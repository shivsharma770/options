#include <ore/research/greeks_validation_study.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <ore/core/market_snapshot.hpp>
#include <ore/core/option.hpp>
#include <ore/core/option_market_snapshot.hpp>
#include <ore/core/provider_greeks.hpp>
#include <ore/core/quote.hpp>
#include <ore/core/types.hpp>
#include <ore/pricing/black_scholes_engine.hpp>
#include <ore/pricing/greeks.hpp>
#include <ore/pricing/pricing_result.hpp>
#include <ore/research/research_context.hpp>
#include <ore/research/research_csv.hpp>
#include <ore/research/research_report.hpp>
#include <ore/research/research_study.hpp>

namespace ore::research {

namespace {

/// True iff the option is priceable *and* worth including in the
/// Greek comparison. See `GreeksValidationStudy::Config` for the
/// individual switches.
[[nodiscard]] bool candidate(const ore::core::OptionMarketSnapshot& oms,
                             std::chrono::year_month_day valuation_date,
                             const GreeksValidationStudy::Config& cfg)
{
    const auto& q   = oms.quote;
    const auto& opt = oms.option;

    if (opt.strike <= 0.0) return false;
    if (opt.exercise != ore::core::ExerciseStyle::European) return false;
    if (std::chrono::sys_days{opt.expiration} < std::chrono::sys_days{valuation_date}) {
        return false;
    }
    if (cfg.require_provider_iv) {
        if (!q.implied_volatility.has_value() || *q.implied_volatility <= 0.0) {
            return false;
        }
    }
    if (!q.provider_greeks.has_value()) return false;

    if (cfg.require_all_provider_greeks && !q.provider_greeks->complete()) {
        return false;
    }
    if (!q.provider_greeks->any()) return false;
    return true;
}

/// Compute the absolute-difference statistics for a single Greek's
/// error vector. Percentiles use nearest-rank on a sorted copy — the
/// standard textbook definition, adequate for typical N ~ 1e6.
GreeksValidationStudy::GreekStats
aggregate(std::vector<double> errors) {
    GreeksValidationStudy::GreekStats out{};
    if (errors.empty()) return out;
    std::sort(errors.begin(), errors.end());
    const auto n = errors.size();

    double sum   = 0.0;
    double sumsq = 0.0;
    double worst = 0.0;
    for (double e : errors) {
        const double a = std::abs(e);
        sum   += a;
        sumsq += a * a;
        if (a > worst) worst = a;
    }
    out.comparisons = n;
    out.mae   = sum / static_cast<double>(n);
    out.rmse  = std::sqrt(sumsq / static_cast<double>(n));
    out.worst = worst;

    auto pct = [&](double p) {
        const std::size_t idx = static_cast<std::size_t>(
            std::min<std::size_t>(n - 1,
                                  static_cast<std::size_t>(p * static_cast<double>(n))));
        return std::abs(errors[idx]);
    };
    out.p50 = pct(0.50);
    out.p95 = pct(0.95);
    out.p99 = pct(0.99);
    return out;
}

/// Populate the (provider, computed, error) triple for one Greek if
/// the provider value is present. `computed` is expected in the
/// vendor's unit convention already (the caller applies the
/// vendor-specific scale factors before this).
void set_triple(std::optional<double>& p_slot,
                std::optional<double>& c_slot,
                std::optional<double>& e_slot,
                std::optional<double> provider,
                double computed)
{
    if (!provider.has_value()) return;
    p_slot = *provider;
    c_slot = computed;
    e_slot = computed - *provider;
}

} // namespace

void GreeksValidationStudy::begin(const ResearchContext& /*ctx*/) {
    rows_.clear();
    stats_ = Statistics{};
    skipped_ = 0;
}

void GreeksValidationStudy::process(const ResearchContext& ctx) {
    std::vector<Row> per_day_rows;
    per_day_rows.reserve(ctx.chain().size());
    std::size_t per_day_skipped = 0;

    const ore::pricing::BlackScholesEngine engine{};
    const auto valuation_date = ctx.date();

    for (const auto& oms : ctx.chain().options()) {
        if (!candidate(oms, valuation_date, config_)) {
            ++per_day_skipped;
            continue;
        }

        const auto& opt = oms.option;
        const auto& q   = oms.quote;

        // Build a market snapshot that carries the provider's IV so
        // BS is evaluated at *the same* volatility the provider used.
        auto market = ctx.market();
        market.volatility = *q.implied_volatility;

        ore::pricing::PricingResult res{};
        try {
            res = engine.price(opt, market);
        } catch (const std::exception&) {
            ++per_day_skipped;
            continue;
        }

        Row row{};
        row.date        = valuation_date;
        row.expiration  = opt.expiration;
        row.strike      = opt.strike;
        row.type        = opt.type;
        row.provider_iv = *q.implied_volatility;

        const auto& pg = *q.provider_greeks;

        set_triple(row.provider_delta, row.computed_delta, row.delta_error,
                   pg.delta, res.greeks.delta);
        set_triple(row.provider_gamma, row.computed_gamma, row.gamma_error,
                   pg.gamma, res.greeks.gamma);
        set_triple(row.provider_theta, row.computed_theta, row.theta_error,
                   pg.theta, res.greeks.theta * config_.theta_scale);
        set_triple(row.provider_vega,  row.computed_vega,  row.vega_error,
                   pg.vega,  res.greeks.vega  * config_.vega_scale);
        set_triple(row.provider_rho,   row.computed_rho,   row.rho_error,
                   pg.rho,   res.greeks.rho);

        per_day_rows.push_back(row);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    rows_.insert(rows_.end(),
                 std::make_move_iterator(per_day_rows.begin()),
                 std::make_move_iterator(per_day_rows.end()));
    skipped_ += per_day_skipped;
}

void GreeksValidationStudy::end(const ResearchContext& ctx, ResearchReport& report) {
    // Collect per-Greek error vectors from the accumulated rows.
    // Each row may contribute to any subset of the five Greeks
    // depending on which provider values were populated.
    std::vector<double> ed, eg, et, ev, er;
    ed.reserve(rows_.size()); eg.reserve(rows_.size());
    et.reserve(rows_.size()); ev.reserve(rows_.size());
    er.reserve(rows_.size());

    for (const auto& r : rows_) {
        if (r.delta_error) ed.push_back(*r.delta_error);
        if (r.gamma_error) eg.push_back(*r.gamma_error);
        if (r.theta_error) et.push_back(*r.theta_error);
        if (r.vega_error)  ev.push_back(*r.vega_error);
        if (r.rho_error)   er.push_back(*r.rho_error);
    }

    stats_.delta = aggregate(std::move(ed));
    stats_.gamma = aggregate(std::move(eg));
    stats_.theta = aggregate(std::move(et));
    stats_.vega  = aggregate(std::move(ev));
    stats_.rho   = aggregate(std::move(er));

    report.processed_contracts = rows_.size();
    report.skipped_contracts   = skipped_;

    const auto csv_path = ctx.output_dir() / config_.output_filename;
    ResearchCsvWriter w(csv_path, {
        "date",
        "expiration",
        "strike",
        "option_type",
        "provider_iv",
        "provider_delta", "computed_delta", "delta_error",
        "provider_gamma", "computed_gamma", "gamma_error",
        "provider_theta", "computed_theta", "theta_error",
        "provider_vega",  "computed_vega",  "vega_error",
        "provider_rho",   "computed_rho",   "rho_error",
    });
    for (const auto& r : rows_) {
        w.write_row({
            ResearchCsvWriter::date(r.date),
            ResearchCsvWriter::date(r.expiration),
            ResearchCsvWriter::number(r.strike),
            std::string(ore::core::to_string(r.type)),
            ResearchCsvWriter::number(r.provider_iv),
            ResearchCsvWriter::optional_number(r.provider_delta),
            ResearchCsvWriter::optional_number(r.computed_delta),
            ResearchCsvWriter::optional_number(r.delta_error),
            ResearchCsvWriter::optional_number(r.provider_gamma),
            ResearchCsvWriter::optional_number(r.computed_gamma),
            ResearchCsvWriter::optional_number(r.gamma_error),
            ResearchCsvWriter::optional_number(r.provider_theta),
            ResearchCsvWriter::optional_number(r.computed_theta),
            ResearchCsvWriter::optional_number(r.theta_error),
            ResearchCsvWriter::optional_number(r.provider_vega),
            ResearchCsvWriter::optional_number(r.computed_vega),
            ResearchCsvWriter::optional_number(r.vega_error),
            ResearchCsvWriter::optional_number(r.provider_rho),
            ResearchCsvWriter::optional_number(r.computed_rho),
            ResearchCsvWriter::optional_number(r.rho_error),
        });
    }

    report.generated_files.push_back(csv_path);
}

std::unique_ptr<ResearchStudy> GreeksValidationStudy::clone() const {
    return std::make_unique<GreeksValidationStudy>(config_);
}

void GreeksValidationStudy::merge(const ResearchStudy& other) {
    const auto* rhs = dynamic_cast<const GreeksValidationStudy*>(&other);
    if (!rhs) return;
    std::lock_guard<std::mutex> lock(mutex_);
    rows_.insert(rows_.end(), rhs->rows_.begin(), rhs->rows_.end());
    skipped_ += rhs->skipped_;
}

} // namespace ore::research
