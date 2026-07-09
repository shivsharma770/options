/**
 * @file option_chain_calibrator.hpp
 * @brief Whole-chain implied-volatility calibration and diagnostics.
 *
 * ### The calibration workflow
 *
 * @code
 *   OptionChain chain  = YahooOptionLoader::load(...);      // market data
 *   OptionChainCalibrator calibrator;                       // stateless
 *   CalibrationReport   report = calibrator.calibrate(chain);
 *   report.write_csv("out.csv");                            // for Python
 * @endcode
 *
 * The calibrator walks every `OptionMarketSnapshot` in the chain and,
 * for each contract:
 *   1. Applies the filter rules (see `SkipReason`) — contracts that fail
 *      filters are recorded but never handed to the solver.
 *   2. Computes the market price as the bid-ask **midpoint**.
 *   3. Invokes `ImpliedVolatilitySolver` to recover sigma from that price.
 *   4. Compares the computed sigma to the provider's IV, when available.
 *   5. Aggregates iteration and error statistics into the report.
 *
 * The result is a `CalibrationReport` containing one `CalibrationResult`
 * per contract plus summary metrics. The report knows how to emit a CSV
 * shaped for downstream Python analysis (`report.write_csv(...)`); the
 * column layout is stable and covers everything needed to build smiles,
 * skews, term structures, and provider-vs-computed error histograms
 * without any further C++ work.
 *
 * ### Why the midpoint?
 *
 * `mid = (bid + ask) / 2` is the standard academic convention for
 * "the market's fair price" when calibrating options:
 *
 *   * The bid alone is the price a market-maker will *buy* at (i.e. the
 *     price you would receive selling into the market); the ask is what
 *     they will *sell* at. Neither in isolation is neutral.
 *   * The mid is the theoretical price at which both sides of the spread
 *     would clear if the spread were infinitesimal; it is also the
 *     price implied by risk-neutral pricing theory.
 *   * Using the last traded price introduces a stale-quote bias: last
 *     trade could be minutes or hours old and no longer reflect current
 *     market state.
 *   * References: Hull (2018) *Options, Futures, and Other Derivatives*,
 *     Ch. 20; Cox and Rubinstein (1985) *Options Markets*.
 *
 * Once we support wider quality analytics we may add mid variants
 * weighted by size, or robust to wide spreads. The plumbing is here.
 *
 * ### Why re-derive IV when the provider publishes one?
 *
 * Because we do not know the provider's model. Yahoo uses a proprietary
 * pricing engine (rumoured to be a binomial tree for American-style
 * equity options), which will *disagree* with our Black-Scholes engine
 * on American names by a systematic amount — the American premium.
 * A large systematic gap between provider IV and our IV is therefore
 * *not* a bug in either implementation; it is the American-vs-European
 * exercise mismatch. Small residual differences on European names
 * (SPX, XSP) are model / rounding artefacts and are the actionable
 * validation signal. See `README.md` and the discussion in
 * `docs/BLACK_SCHOLES_VALIDATION.md`.
 */
#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include <ore/core/option.hpp>
#include <ore/core/option_chain.hpp>
#include <ore/numerics/solver_result.hpp>
#include <ore/pricing/implied_volatility_solver.hpp>

namespace ore::analytics {

/**
 * @brief Reason a contract was not handed to the solver.
 *
 * A contract that clears every filter has `SkipReason::None` and is
 * *attempted* — whether the attempt converged is separately recorded
 * in `CalibrationResult::solver_status`.
 */
enum class SkipReason {
    /** Not skipped — the contract was calibrated (may or may not have converged). */
    None,

    /**
     * `bid == 0 && ask == 0`. Yahoo / OCC convention for "no market".
     * Deep-OTM contracts are listed but not quoted; there is no price
     * to invert.
     */
    NoMarket,

    /** `bid > ask` with both positive. Crossed markets are typically stale
     *  or feed errors — the loader already rejects these but we defensively
     *  re-check in case a chain is constructed by other means. */
    CrossedMarket,

    /** Any of `bid`, `ask`, `mid` is `NaN` or `Inf`. Data corruption. */
    NonFiniteQuote,

    /** `mid <= 0`. Non-positive midpoint despite passing the finite-quote
     *  check — no valid option price is <= 0 by the model. */
    NonPositiveMidPrice,

    /** `mid < config.minimum_option_price`. Sub-tick prices carry roughly
     *  one tick of information; the implied vol is effectively unresolvable
     *  and dominated by rounding noise. */
    BelowMinimumPrice,

    /** Contract has already expired at the snapshot's valuation date.
     *  The loader rejects these; belt-and-braces. */
    Expired,

    /** `mid` lies outside the theoretical Black-Scholes arbitrage bounds
     *  `[max(0, disc_S - disc_K), disc_S]` for calls (put analog for
     *  puts). Solving would either throw or return the degenerate σ=0
     *  edge case; skipping preserves the "not calibrated" semantics. */
    ArbitrageViolation,
};

/** Human-readable name of a `SkipReason` for CSV export and logging. */
[[nodiscard]] constexpr std::string_view to_string(SkipReason r) noexcept {
    switch (r) {
        case SkipReason::None:                 return "None";
        case SkipReason::NoMarket:             return "NoMarket";
        case SkipReason::CrossedMarket:        return "CrossedMarket";
        case SkipReason::NonFiniteQuote:       return "NonFiniteQuote";
        case SkipReason::NonPositiveMidPrice:  return "NonPositiveMidPrice";
        case SkipReason::BelowMinimumPrice:    return "BelowMinimumPrice";
        case SkipReason::Expired:              return "Expired";
        case SkipReason::ArbitrageViolation:   return "ArbitrageViolation";
    }
    return "Unknown";
}

/**
 * @brief Per-contract calibration outcome.
 *
 * One `CalibrationResult` is emitted for every option in the input
 * chain — including those we skipped. A CSV consumer can filter down to
 * `skip_reason == None && solver_status == Converged` for the "clean"
 * subset used to build smiles / surfaces.
 */
struct CalibrationResult {
    /** Provider-assigned identifier (e.g. `"SPY260808C00470000"`). */
    std::string contract_symbol;

    /** Contract terms — strike, expiration, type, exercise. */
    ore::core::Option option{};

    /** Bid at the snapshot. */
    double bid{0.0};
    /** Ask at the snapshot. */
    double ask{0.0};
    /** Last traded price at the snapshot. */
    double last{0.0};
    /** Midpoint `0.5 * (bid + ask)` — the market price we handed to the
     *  solver. Populated even for skipped contracts (as a diagnostic),
     *  provided the raw quotes are finite. */
    double mid_price{0.0};

    /** Provider-published implied volatility. `std::nullopt` when the
     *  provider does not publish this contract's IV (deep OTM, thin
     *  markets). */
    std::optional<double> provider_iv{};

    /** Our computed implied volatility, when the solver converged.
     *  `std::nullopt` when the solver did not converge or the contract
     *  was skipped. */
    std::optional<double> computed_iv{};

    /** How the solver terminated. Meaningful only for non-skipped
     *  contracts; skipped contracts leave this at the default. */
    ore::numerics::SolverStatus solver_status{
        ore::numerics::SolverStatus::MaxIterationsReached
    };

    /** Iterations consumed by the winning method (see
     *  `ImpliedVolatilityResult::iterations`). Zero for skipped
     *  contracts. */
    std::size_t iterations{0};

    /** True iff the bisection fallback was invoked. Meaningful only for
     *  non-skipped contracts. */
    bool used_bisection{false};

    /** Price residual `|BS(computed_iv) - mid|` at termination. Small on
     *  convergence, large otherwise. */
    double solver_residual{0.0};

    /** `|provider_iv - computed_iv|`, when both exist. */
    std::optional<double> absolute_error{};

    /** `|provider_iv - computed_iv| / |provider_iv|`, when both exist
     *  and `provider_iv` is non-zero. */
    std::optional<double> relative_error{};

    /** Why the contract was skipped, or `None` if it was calibrated. */
    SkipReason skip_reason{SkipReason::None};

    /** `true` iff the contract was actually calibrated *and* the solver
     *  converged. */
    [[nodiscard]] bool was_calibrated() const noexcept {
        return skip_reason == SkipReason::None
            && solver_status == ore::numerics::SolverStatus::Converged;
    }

    /** `true` iff the contract was filtered out before invoking the solver. */
    [[nodiscard]] bool was_skipped() const noexcept {
        return skip_reason != SkipReason::None;
    }
};

/**
 * @brief Aggregate outcome of calibrating a full chain.
 *
 * The `results` vector always has size equal to the input chain
 * (`chain.size()`); each entry corresponds one-for-one to the chain's
 * option ordering. Summary metrics are computed *only* over the relevant
 * subsets:
 *   - iteration statistics: over successful solves only
 *   - IV-error statistics:  over contracts with both provider *and*
 *     computed IV
 *
 * Consumers building smiles / surfaces should filter to
 * `r.was_calibrated()`. Consumers doing provider-agreement histograms
 * should filter to `r.absolute_error.has_value()`.
 */
struct CalibrationReport {
    /** One result per option, in chain order. */
    std::vector<CalibrationResult> results{};

    /** Contracts whose solver reported `SolverStatus::Converged`. */
    std::size_t successful_solves{0};

    /** Contracts whose solver ran but did *not* converge. */
    std::size_t failed_solves{0};

    /** Contracts skipped before invoking the solver (any `SkipReason != None`). */
    std::size_t skipped{0};

    /** Contracts where `ImpliedVolatilityMethod == Bisection`. Subset of
     *  `successful_solves` in practice (Newton failed, bisection saved
     *  it), but bisection can itself fail — the count includes those
     *  failures too. */
    std::size_t bisection_fallbacks{0};

    /** Arithmetic mean of `iterations` over `successful_solves`. Zero
     *  if there were none. */
    double average_iterations{0.0};

    /** Maximum `iterations` seen over successful solves. Zero if none. */
    std::size_t maximum_iterations{0};

    /** Number of contracts contributing to the IV-error statistics
     *  (both provider *and* computed IV present). */
    std::size_t provider_iv_comparisons{0};

    /** Arithmetic mean of `|provider_iv - computed_iv|`. Zero if no
     *  comparisons. */
    double mean_absolute_iv_error{0.0};

    /** Root-mean-square of `provider_iv - computed_iv`. Zero if no
     *  comparisons. */
    double rmse_iv_error{0.0};

    /** Maximum `|provider_iv - computed_iv|`. Zero if no comparisons. */
    double maximum_iv_error{0.0};

    /**
     * Fraction of *attempted* solves that converged. Denominator is
     * `successful_solves + failed_solves` — skipped contracts are not
     * counted (we never attempted them). Returns 0 if there were no
     * attempts (denominator would be zero).
     */
    [[nodiscard]] double convergence_rate() const noexcept {
        const auto attempted = successful_solves + failed_solves;
        if (attempted == 0) return 0.0;
        return static_cast<double>(successful_solves)
             / static_cast<double>(attempted);
    }

    /**
     * @brief Emit the report as CSV, one row per input contract.
     *
     * Columns (in order):
     *   `contract_symbol, expiration, strike, option_type, bid, ask,
     *    last, mid_price, provider_iv, computed_iv, absolute_error,
     *    relative_error, solver_status, iterations, used_bisection,
     *    solver_residual, skip_reason`.
     *
     * Missing optional values are written as empty fields (pandas reads
     * them as `NaN`). Booleans are written as `0` / `1`. Dates are ISO
     * 8601 (`YYYY-MM-DD`), matching the loader's convention. Numeric
     * fields are written with `%.17g` (round-trip-preserving); tests
     * that parse the CSV back must do so with matching precision.
     *
     * The CSV shape is deliberately stable across milestones: adding
     * columns is a breaking change, and future versions will append
     * only.
     */
    void write_csv(std::ostream& os) const;

    /** Convenience overload that writes to a file path. */
    void write_csv(const std::filesystem::path& path) const;
};

/**
 * @class OptionChainCalibrator
 * @brief Calibrates every option in an `OptionChain` to its market
 *        midpoint and reports diagnostics.
 *
 * Stateless — safe to reuse across chains and threads. Owns a default
 * `ImpliedVolatilitySolver` internally; supply a custom `Config` to
 * tighten tolerances, disable bisection fallback, or supply an initial-
 * guess override for all contracts.
 */
class OptionChainCalibrator {
public:
    /**
     * @brief Configuration knobs.
     *
     * Splits cleanly into two groups:
     *   - `solver_config`      : passed through to `ImpliedVolatilitySolver`.
     *   - `minimum_option_price`: filter threshold for the calibrator.
     */
    struct Config {
        /** Configuration for the underlying implied-volatility solver.
         *  Its default is fine for SPY-style equity chains. */
        ore::pricing::ImpliedVolatilitySolver::Config solver_config{};

        /** Contracts with `mid < minimum_option_price` are skipped with
         *  `SkipReason::BelowMinimumPrice`. Default of `$0.005` matches
         *  half of an equity-options penny tick — below this the price
         *  is essentially noise. Set to `0.0` to disable this filter. */
        double minimum_option_price{0.005};
    };

    OptionChainCalibrator() = default;

    /** Construct with a specific configuration. */
    explicit OptionChainCalibrator(Config config) : config_(config) {}

    /** Access the immutable configuration. */
    [[nodiscard]] const Config& config() const noexcept { return config_; }

    /**
     * @brief Calibrate every option in the chain and aggregate results.
     *
     * Never throws for individual-contract issues — the state is
     * reported in `CalibrationResult::skip_reason` or
     * `.solver_status`. May throw `std::invalid_argument` if the input
     * chain itself is malformed (empty required fields, non-finite
     * market snapshot).
     */
    [[nodiscard]] CalibrationReport calibrate(
        const ore::core::OptionChain& chain) const;

private:
    Config config_{};
};

} // namespace ore::analytics
