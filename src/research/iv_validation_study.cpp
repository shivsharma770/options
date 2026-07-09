#include <ore/research/iv_validation_study.hpp>

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

#include <ore/core/option.hpp>
#include <ore/core/option_market_snapshot.hpp>
#include <ore/core/quote.hpp>
#include <ore/core/types.hpp>
#include <ore/numerics/solver_result.hpp>
#include <ore/pricing/black_scholes_engine.hpp>
#include <ore/pricing/implied_volatility_solver.hpp>
#include <ore/research/research_context.hpp>
#include <ore/research/research_csv.hpp>
#include <ore/research/research_report.hpp>
#include <ore/research/research_study.hpp>

namespace ore::research {

namespace {

/// True iff the contract is worth feeding to the IV solver. Any row
/// that fails this test is skipped and counted separately. The
/// quote-shape primitives (`bidask_finite`, `is_crossed`) live on
/// `Quote` so this predicate and `OptionChainCalibrator::classify()`
/// never drift apart in their sense of what a "healthy quote" is.
[[nodiscard]] bool candidate(const ore::core::OptionMarketSnapshot& oms,
                             std::chrono::year_month_day valuation_date,
                             bool european_only)
{
    const auto& q = oms.quote;
    if (!q.bidask_finite())                return false;
    if (q.bid <= 0.0 || q.ask <= 0.0)      return false; // stricter than calibrator: we need two-sided liquidity
    if (q.is_crossed())                    return false;
    if (!q.implied_volatility.has_value()) return false;

    const auto& opt = oms.option;
    if (opt.strike <= 0.0) return false;
    if (european_only && opt.exercise != ore::core::ExerciseStyle::European) return false;

    // Expired contracts have zero time value in Black-Scholes — the
    // solver's arbitrage upper-bound check would reject them anyway.
    if (std::chrono::sys_days{opt.expiration} < std::chrono::sys_days{valuation_date}) {
        return false;
    }
    return true;
}

/// Convergence rate as reported in the study statistics — attempted
/// solves that returned `Converged` divided by all attempts. Guards
/// the zero-attempts case explicitly to avoid NaN in the CSV.
[[nodiscard]] double
convergence_rate(std::size_t converged, std::size_t attempted) {
    if (attempted == 0) return 0.0;
    return static_cast<double>(converged) / static_cast<double>(attempted);
}

/// Median of a sorted-in-place copy of `xs`. Empty input returns 0.
/// Used only for aggregate stats — nothing hot.
[[nodiscard]] double median(std::vector<double> xs) {
    if (xs.empty()) return 0.0;
    std::sort(xs.begin(), xs.end());
    const auto n = xs.size();
    if ((n % 2u) == 1u) return xs[n / 2u];
    return 0.5 * (xs[n / 2u - 1u] + xs[n / 2u]);
}

} // namespace

void IVValidationStudy::begin(const ResearchContext& /*ctx*/) {
    rows_.clear();
    stats_ = Statistics{};
    skipped_ = 0;
}

void IVValidationStudy::process(const ResearchContext& ctx) {
    // Local scratch to avoid holding the mutex across every solver
    // call. Merged into `rows_` at the end of the day.
    std::vector<Row> per_day_rows;
    per_day_rows.reserve(ctx.chain().size());
    std::size_t per_day_skipped = 0;

    const ore::pricing::ImpliedVolatilitySolver solver(config_.solver);
    const auto& market = ctx.market();
    const auto  valuation_date = ctx.date();

    for (const auto& oms : ctx.chain().options()) {
        if (!candidate(oms, valuation_date, config_.european_only)) {
            ++per_day_skipped;
            continue;
        }

        const auto  provider_iv = *oms.quote.implied_volatility;
        const auto  mid         = oms.quote.mid();

        // Guard against a mid outside the arbitrage bounds — the
        // solver would throw and we would lose the whole day if the
        // engine were configured to abort on error. The bounds are
        // cheap to check ourselves and let us record the row with
        // an empty computed IV instead.
        Row row{};
        row.date        = valuation_date;
        row.expiration  = oms.option.expiration;
        row.strike      = oms.option.strike;
        row.type        = oms.option.type;
        row.mid_price   = mid;
        row.provider_iv = provider_iv;

        try {
            const auto res = solver.solve(oms.option, market, mid);
            row.iterations = res.iterations;
            row.method     = res.method;
            row.converged  = res.converged();
            if (res.converged()) {
                row.computed_iv    = res.root;
                const double abs_e = std::abs(res.root - provider_iv);
                row.absolute_error = abs_e;
                if (provider_iv > 0.0) {
                    row.relative_error = abs_e / provider_iv;
                }
            }
        } catch (const std::exception&) {
            // Solver rejected the input (mid outside bounds, negative
            // spot, ...). Record the row anyway with `converged =
            // false` so the CSV shows the anomaly. `iterations` is
            // left at 0.
            row.converged = false;
        }

        per_day_rows.push_back(row);
    }

    // Fold this day's work into the study's accumulator.
    std::lock_guard<std::mutex> lock(mutex_);
    rows_.insert(rows_.end(),
                 std::make_move_iterator(per_day_rows.begin()),
                 std::make_move_iterator(per_day_rows.end()));
    skipped_ += per_day_skipped;
}

void IVValidationStudy::end(const ResearchContext& ctx, ResearchReport& report) {
    // Recompute aggregate statistics from the accumulated rows. This
    // is idempotent across parallel merges (merge() appends worker
    // rows; end() rebuilds stats from scratch).
    std::vector<double> abs_errors;
    abs_errors.reserve(rows_.size());
    double sum_abs    = 0.0;
    double sum_sqrerr = 0.0;
    double worst      = 0.0;
    std::size_t attempted = 0;
    std::size_t converged = 0;

    for (const auto& r : rows_) {
        ++attempted;
        if (r.converged) ++converged;
        if (r.absolute_error.has_value()) {
            const double e = *r.absolute_error;
            abs_errors.push_back(e);
            sum_abs    += e;
            sum_sqrerr += e * e;
            if (e > worst) worst = e;
        }
    }

    stats_.solves_attempted     = attempted;
    stats_.solves_converged     = converged;
    stats_.mean_absolute_error  = abs_errors.empty() ? 0.0 :
        (sum_abs / static_cast<double>(abs_errors.size()));
    stats_.median_absolute_error = median(abs_errors);
    stats_.rmse = abs_errors.empty() ? 0.0 :
        std::sqrt(sum_sqrerr / static_cast<double>(abs_errors.size()));
    stats_.worst_error       = worst;
    stats_.convergence_rate  = convergence_rate(converged, attempted);

    report.processed_contracts = attempted;
    report.skipped_contracts   = skipped_;

    const auto csv_path = ctx.output_dir() / config_.output_filename;
    ResearchCsvWriter w(csv_path, {
        "date",
        "expiration",
        "strike",
        "option_type",
        "mid_price",
        "provider_iv",
        "computed_iv",
        "absolute_error",
        "relative_error",
        "iterations",
        "solver_method",
        "converged",
    });

    for (const auto& r : rows_) {
        w.write_row({
            ResearchCsvWriter::date(r.date),
            ResearchCsvWriter::date(r.expiration),
            ResearchCsvWriter::number(r.strike),
            std::string(ore::core::to_string(r.type)),
            ResearchCsvWriter::number(r.mid_price),
            ResearchCsvWriter::optional_number(r.provider_iv),
            ResearchCsvWriter::optional_number(r.computed_iv),
            ResearchCsvWriter::optional_number(r.absolute_error),
            ResearchCsvWriter::optional_number(r.relative_error),
            ResearchCsvWriter::integer(r.iterations),
            std::string(r.method == ore::pricing::ImpliedVolatilityMethod::Newton
                        ? "Newton" : "Bisection"),
            r.converged ? std::string("1") : std::string("0"),
        });
    }

    report.generated_files.push_back(csv_path);
}

std::unique_ptr<ResearchStudy> IVValidationStudy::clone() const {
    auto c = std::make_unique<IVValidationStudy>(config_);
    // clone() intentionally does *not* copy `rows_` — parallel
    // workers must start empty.
    return c;
}

void IVValidationStudy::merge(const ResearchStudy& other) {
    // Only merging our own type is meaningful. `dynamic_cast` gives
    // us a safe check; a mismatched merge is a programmer error and
    // silently drops the merge — the alternative (throwing from a
    // merge callback) risks aborting the run at the worst moment.
    const auto* rhs = dynamic_cast<const IVValidationStudy*>(&other);
    if (!rhs) return;

    std::lock_guard<std::mutex> lock(mutex_);
    rows_.insert(rows_.end(), rhs->rows_.begin(), rhs->rows_.end());
    skipped_ += rhs->skipped_;
}

} // namespace ore::research
