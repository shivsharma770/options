#include <ore/analytics/option_chain_calibrator.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <ios>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <ore/pricing/black_scholes_engine.hpp>

namespace ore::analytics {

namespace {

using ore::core::MarketSnapshot;
using ore::core::Option;
using ore::core::OptionChain;
using ore::core::OptionMarketSnapshot;
using ore::core::OptionType;
using ore::core::Quote;
using ore::numerics::SolverStatus;
using ore::pricing::BlackScholesEngine;
using ore::pricing::ImpliedVolatilityMethod;
using ore::pricing::ImpliedVolatilitySolver;

/**
 * Time-to-expiry in years, using the same ACT/365F convention as the
 * Black-Scholes engine (`(days_between / 365.0)`).
 */
double year_fraction(const std::chrono::year_month_day& valuation,
                     const std::chrono::year_month_day& expiration) noexcept
{
    const auto val_days = std::chrono::sys_days{valuation};
    const auto exp_days = std::chrono::sys_days{expiration};
    return static_cast<double>((exp_days - val_days).count()) / 365.0;
}

/**
 * Theoretical Black-Scholes arbitrage bounds for a European option
 * price. Below the lower bound the price contradicts the sigma-monotone
 * BS formula; above the upper bound it contradicts no-arbitrage.
 */
struct ArbitrageBounds {
    double lower;
    double upper;
};

ArbitrageBounds bounds(const BlackScholesEngine::Inputs& in) noexcept {
    const double disc_S = in.spot   * std::exp(-in.dividend_yield * in.time_to_expiry);
    const double disc_K = in.strike * std::exp(-in.rate           * in.time_to_expiry);
    if (in.type == OptionType::Call) {
        return {std::max(0.0, disc_S - disc_K), disc_S};
    }
    return {std::max(0.0, disc_K - disc_S), disc_K};
}

/**
 * Apply the calibrator's filter rules and return the reason to skip, or
 * `SkipReason::None` if the contract should be handed to the solver.
 *
 * The order of checks matters: bad quotes are diagnosed *before*
 * arbitrage-bound checks, because arbitrage bounds are only meaningful
 * for finite, positive prices.
 */
SkipReason classify(
    const Quote& q,
    double mid,
    const BlackScholesEngine::Inputs& in,
    const OptionChainCalibrator::Config& cfg) noexcept
{
    // The `Quote::bidask_finite()`, `is_unquoted()`, and
    // `is_crossed()` primitives centralise the "quote well-formed"
    // vocabulary shared with the research studies; this function
    // still owns the calibrator-specific SkipReason mapping and the
    // arbitrage-bounds check.
    if (in.time_to_expiry <= 0.0)              return SkipReason::Expired;
    if (!q.bidask_finite() || !std::isfinite(mid))
                                                return SkipReason::NonFiniteQuote;
    if (q.is_unquoted())                        return SkipReason::NoMarket;
    if (q.is_crossed() && q.bid > 0.0)          return SkipReason::CrossedMarket;
    if (mid <= 0.0)                             return SkipReason::NonPositiveMidPrice;
    if (mid < cfg.minimum_option_price)         return SkipReason::BelowMinimumPrice;

    const auto b = bounds(in);
    if (mid > b.upper || mid < b.lower)         return SkipReason::ArbitrageViolation;

    return SkipReason::None;
}

/**
 * Assemble a `CalibrationResult` for a contract we did not calibrate.
 * The skip reason is recorded but no solver was invoked; provider
 * comparison fields remain empty.
 */
CalibrationResult skipped_result(
    const OptionMarketSnapshot& snap,
    double mid,
    SkipReason reason)
{
    CalibrationResult r{};
    r.contract_symbol = snap.contract_symbol;
    r.option          = snap.option;
    r.bid             = snap.quote.bid;
    r.ask             = snap.quote.ask;
    r.last            = snap.quote.last;
    r.mid_price       = mid;
    r.provider_iv     = snap.quote.implied_volatility;
    r.skip_reason     = reason;
    return r;
}

/**
 * Compute absolute + relative error between provider and computed IV,
 * *if* both exist. Relative error uses `|provider_iv|` as the denominator
 * (standard convention). When the provider reports exactly zero IV we
 * populate `absolute_error` but leave `relative_error` empty rather than
 * dividing by zero.
 */
void populate_errors(CalibrationResult& r) noexcept {
    if (!r.provider_iv.has_value() || !r.computed_iv.has_value()) return;
    const double abs_err = std::abs(*r.provider_iv - *r.computed_iv);
    r.absolute_error = abs_err;
    if (*r.provider_iv != 0.0) {
        r.relative_error = abs_err / std::abs(*r.provider_iv);
    }
}

/**
 * Format a double for CSV output. `%.17g` is round-trip-preserving under
 * IEEE 754 rules — the same value read back gives the same bits. Non-
 * finite values (`NaN` / `Inf`) become empty fields so pandas reads them
 * as `NaN`; this matches the loader's tolerant-parsing convention.
 */
std::string format_double(double x) {
    if (!std::isfinite(x)) return {};
    char buf[32];
    const int n = std::snprintf(buf, sizeof(buf), "%.17g", x);
    if (n <= 0) return {};
    return std::string(buf, buf + n);
}

std::string format_optional(const std::optional<double>& v) {
    return v.has_value() ? format_double(*v) : std::string{};
}

/**
 * ISO 8601 date (`YYYY-MM-DD`) — matches the loader's format so the CSVs
 * are re-loadable if we ever want that. `std::format` would be cleaner
 * but not universally available on our supported toolchains yet, so we
 * hand-roll.
 */
std::string format_date(const std::chrono::year_month_day& ymd) {
    char buf[16];
    const int y = static_cast<int>(ymd.year());
    const unsigned m = static_cast<unsigned>(ymd.month());
    const unsigned d = static_cast<unsigned>(ymd.day());
    const int n = std::snprintf(buf, sizeof(buf), "%04d-%02u-%02u", y, m, d);
    if (n <= 0) return {};
    return std::string(buf, buf + n);
}

std::string_view type_string(OptionType t) noexcept {
    return t == OptionType::Call ? "call" : "put";
}

}  // namespace

CalibrationReport OptionChainCalibrator::calibrate(const OptionChain& chain) const {
    const MarketSnapshot& market = chain.market();
    ImpliedVolatilitySolver solver(config_.solver_config);

    CalibrationReport report{};
    report.results.reserve(chain.size());

    // Running accumulators. We compute the final averages / RMSE at the
    // end to keep the loop body compact.
    std::size_t iterations_sum         = 0;
    double      abs_error_sum          = 0.0;
    double      squared_error_sum      = 0.0;

    for (const auto& snap : chain) {
        const Quote& q = snap.quote;
        const double mid = q.mid();

        const double T = year_fraction(market.valuation_date, snap.option.expiration);
        const BlackScholesEngine::Inputs bs_inputs{
            .spot           = market.spot,
            .strike         = snap.option.strike,
            .rate           = market.risk_free_rate,
            .dividend_yield = market.dividend_yield,
            .volatility     = 0.0, // solved for
            .time_to_expiry = T,
            .type           = snap.option.type,
        };

        const SkipReason reason = classify(q, mid, bs_inputs, config_);
        if (reason != SkipReason::None) {
            report.results.push_back(skipped_result(snap, mid, reason));
            report.skipped++;
            continue;
        }

        // ---- Run the IV solver ---------------------------------------------
        //
        // We call the raw-`Inputs` overload rather than the (Option,
        // MarketSnapshot) one for two reasons:
        //   1. We have already validated the arbitrage bounds here.
        //   2. We have already computed `T`; passing it directly avoids
        //      recomputing the day-count difference inside the solver.
        // Both paths use ACT/365F, so the numerics are identical.
        const auto iv = solver.solve(bs_inputs, mid);

        CalibrationResult cr{};
        cr.contract_symbol = snap.contract_symbol;
        cr.option          = snap.option;
        cr.bid             = q.bid;
        cr.ask             = q.ask;
        cr.last            = q.last;
        cr.mid_price       = mid;
        cr.provider_iv     = q.implied_volatility;
        cr.solver_status   = iv.status;
        cr.iterations      = iv.iterations;
        cr.used_bisection  = (iv.method == ImpliedVolatilityMethod::Bisection);
        cr.solver_residual = iv.residual;
        cr.skip_reason     = SkipReason::None;

        if (iv.converged()) {
            cr.computed_iv = iv.root;
            report.successful_solves++;

            iterations_sum += iv.iterations;
            if (iv.iterations > report.maximum_iterations) {
                report.maximum_iterations = iv.iterations;
            }
        } else {
            report.failed_solves++;
        }
        if (cr.used_bisection) {
            report.bisection_fallbacks++;
        }

        populate_errors(cr);
        if (cr.absolute_error.has_value()) {
            report.provider_iv_comparisons++;
            const double e = *cr.absolute_error;
            abs_error_sum     += e;
            squared_error_sum += e * e;
            if (e > report.maximum_iv_error) {
                report.maximum_iv_error = e;
            }
        }

        report.results.push_back(std::move(cr));
    }

    // ---- Finalize aggregates ----------------------------------------------
    if (report.successful_solves > 0) {
        report.average_iterations = static_cast<double>(iterations_sum)
                                  / static_cast<double>(report.successful_solves);
    }
    if (report.provider_iv_comparisons > 0) {
        const auto n = static_cast<double>(report.provider_iv_comparisons);
        report.mean_absolute_iv_error = abs_error_sum / n;
        report.rmse_iv_error          = std::sqrt(squared_error_sum / n);
    }

    return report;
}

// -----------------------------------------------------------------------------
// CSV export
// -----------------------------------------------------------------------------

void CalibrationReport::write_csv(std::ostream& os) const {
    // Column layout — see the header docstring for stability guarantees.
    os << "contract_symbol,expiration,strike,option_type,"
          "bid,ask,last,mid_price,"
          "provider_iv,computed_iv,absolute_error,relative_error,"
          "solver_status,iterations,used_bisection,solver_residual,"
          "skip_reason\n";

    for (const auto& r : results) {
        os << r.contract_symbol << ','
           << format_date(r.option.expiration) << ','
           << format_double(r.option.strike)   << ','
           << type_string(r.option.type)        << ','
           << format_double(r.bid)              << ','
           << format_double(r.ask)              << ','
           << format_double(r.last)             << ','
           << format_double(r.mid_price)        << ','
           << format_optional(r.provider_iv)    << ','
           << format_optional(r.computed_iv)    << ','
           << format_optional(r.absolute_error) << ','
           << format_optional(r.relative_error) << ','
           << ore::numerics::to_string(r.solver_status) << ','
           << r.iterations                      << ','
           << (r.used_bisection ? 1 : 0)        << ','
           << format_double(r.solver_residual)  << ','
           << ore::analytics::to_string(r.skip_reason)
           << '\n';
    }
}

void CalibrationReport::write_csv(const std::filesystem::path& path) const {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "CalibrationReport::write_csv: failed to open output path: "
            + path.string());
    }
    write_csv(out);
}

}  // namespace ore::analytics
