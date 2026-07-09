#include <ore/research/pricing_validation_study.hpp>

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
#include <ore/core/quote.hpp>
#include <ore/core/types.hpp>
#include <ore/pricing/black_scholes_engine.hpp>
#include <ore/pricing/implied_volatility_solver.hpp>
#include <ore/pricing/pricing_result.hpp>
#include <ore/research/research_context.hpp>
#include <ore/research/research_csv.hpp>
#include <ore/research/research_report.hpp>
#include <ore/research/research_study.hpp>

namespace ore::research {

namespace {

[[nodiscard]] bool candidate(const ore::core::OptionMarketSnapshot& oms,
                             std::chrono::year_month_day valuation_date,
                             const PricingValidationStudy::Config& cfg)
{
    const auto& q   = oms.quote;
    const auto& opt = oms.option;
    if (opt.strike <= 0.0) return false;
    if (opt.exercise != ore::core::ExerciseStyle::European) return false;
    if (std::chrono::sys_days{opt.expiration} < std::chrono::sys_days{valuation_date}) {
        return false;
    }
    if (q.bid <= 0.0 || q.ask <= 0.0) return false;
    if (q.bid > q.ask) return false;
    if (!std::isfinite(q.bid) || !std::isfinite(q.ask)) return false;
    if (cfg.require_provider_iv &&
        (!q.implied_volatility.has_value() || *q.implied_volatility <= 0.0)) {
        return false;
    }
    return true;
}

} // namespace

void PricingValidationStudy::begin(const ResearchContext& /*ctx*/) {
    rows_.clear();
    stats_ = Statistics{};
    skipped_ = 0;
}

void PricingValidationStudy::process(const ResearchContext& ctx) {
    std::vector<Row> per_day_rows;
    per_day_rows.reserve(ctx.chain().size());
    std::size_t per_day_skipped = 0;

    const ore::pricing::ImpliedVolatilitySolver solver(config_.solver);
    const ore::pricing::BlackScholesEngine      engine{};
    const auto valuation_date = ctx.date();
    const auto& market_base   = ctx.market();

    for (const auto& oms : ctx.chain().options()) {
        if (!candidate(oms, valuation_date, config_)) {
            ++per_day_skipped;
            continue;
        }

        const auto& opt = oms.option;
        const auto  mid = oms.quote.mid();

        Row row{};
        row.date       = valuation_date;
        row.expiration = opt.expiration;
        row.strike     = opt.strike;
        row.type       = opt.type;
        row.mid_price  = mid;

        try {
            const auto ivres = solver.solve(opt, market_base, mid);
            row.converged = ivres.converged();
            if (ivres.converged()) {
                auto market = market_base;
                market.volatility = ivres.root;
                const auto pr = engine.price(opt, market);
                row.recovered_iv = ivres.root;
                row.repriced     = pr.price;
                row.residual     = pr.price - mid;
            }
        } catch (const std::exception&) {
            row.converged = false;
        }

        per_day_rows.push_back(row);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    rows_.insert(rows_.end(),
                 std::make_move_iterator(per_day_rows.begin()),
                 std::make_move_iterator(per_day_rows.end()));
    skipped_ += per_day_skipped;
}

void PricingValidationStudy::end(const ResearchContext& ctx, ResearchReport& report) {
    // Aggregate stats over the accumulated rows. Rows without a
    // residual (solver did not converge) are counted in `total`
    // but skipped in the sums — otherwise a single non-convergent
    // row can dominate RMSE.
    double sum_abs  = 0.0;
    double sumsq    = 0.0;
    double worst    = 0.0;
    std::size_t within = 0;
    std::size_t counted = 0;
    for (const auto& r : rows_) {
        if (!r.residual.has_value()) continue;
        const double a = std::abs(*r.residual);
        sum_abs += a;
        sumsq   += a * a;
        if (a > worst) worst = a;
        if (a <= config_.residual_tolerance) ++within;
        ++counted;
    }

    stats_.total              = rows_.size();
    stats_.within_tolerance   = within;
    stats_.mean_abs_residual  = counted == 0 ? 0.0 :
        sum_abs / static_cast<double>(counted);
    stats_.rmse_residual      = counted == 0 ? 0.0 :
        std::sqrt(sumsq / static_cast<double>(counted));
    stats_.worst_abs_residual = worst;
    stats_.fraction_within_tolerance = stats_.total == 0 ? 0.0 :
        static_cast<double>(within) / static_cast<double>(stats_.total);

    report.processed_contracts = rows_.size();
    report.skipped_contracts   = skipped_;

    const auto csv_path = ctx.output_dir() / config_.output_filename;
    ResearchCsvWriter w(csv_path, {
        "date",
        "expiration",
        "strike",
        "option_type",
        "mid_price",
        "recovered_iv",
        "repriced",
        "residual",
        "converged",
    });
    for (const auto& r : rows_) {
        w.write_row({
            ResearchCsvWriter::date(r.date),
            ResearchCsvWriter::date(r.expiration),
            ResearchCsvWriter::number(r.strike),
            std::string(ore::core::to_string(r.type)),
            ResearchCsvWriter::number(r.mid_price),
            ResearchCsvWriter::optional_number(r.recovered_iv),
            ResearchCsvWriter::optional_number(r.repriced),
            ResearchCsvWriter::optional_number(r.residual),
            r.converged ? std::string("1") : std::string("0"),
        });
    }
    report.generated_files.push_back(csv_path);
}

std::unique_ptr<ResearchStudy> PricingValidationStudy::clone() const {
    return std::make_unique<PricingValidationStudy>(config_);
}

void PricingValidationStudy::merge(const ResearchStudy& other) {
    const auto* rhs = dynamic_cast<const PricingValidationStudy*>(&other);
    if (!rhs) return;
    std::lock_guard<std::mutex> lock(mutex_);
    rows_.insert(rows_.end(), rhs->rows_.begin(), rhs->rows_.end());
    skipped_ += rhs->skipped_;
}

} // namespace ore::research
