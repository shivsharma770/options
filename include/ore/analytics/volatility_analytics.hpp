/**
 * @file volatility_analytics.hpp
 * @brief Volatility smiles, term structure, surface, skew metrics, and IV
 *        summary statistics over a completed `CalibrationReport`.
 *
 * ### Workflow
 *
 * @code
 *   OptionChain chain   = YahooOptionLoader::load(...);
 *   CalibrationReport r = OptionChainCalibrator{}.calibrate(chain);
 *
 *   const auto smiles  = build_smiles(r, chain.market());
 *   const auto term    = build_term_structure(r, chain.market());
 *   const auto surface = build_surface(r, chain.market());
 *   const auto skews   = compute_skew_metrics(smiles, chain.market());
 * @endcode
 *
 * All produced structures know how to emit themselves as CSV
 * (`write_csv(std::ostream&)`) in the *long format* pandas expects. See
 * `docs/VOLATILITY_ANALYTICS.md` for the theoretical background and the
 * `python/plot_*.py` scripts for reference visualisations.
 *
 * ### Design principles
 *
 *  - **No interpolation of the surface itself.** Missing grid cells are
 *    left as `NaN`. Interpolation and parametric surface fitting belong
 *    to the next milestone.
 *  - **Data preservation.** Smiles include *both* calls and puts at the
 *    same strike when both were calibrated. A parallel `types` vector
 *    lets consumers filter (e.g. OTM-only) themselves.
 *  - **Only calibrated rows are consumed.** Skipped or non-converged
 *    contracts are ignored — their IV is either unknown or unreliable.
 *  - **Free-function API.** These operations are pure transformations
 *    with no per-run state; classes would only add noise.
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <ostream>
#include <span>
#include <string_view>
#include <vector>

#include <ore/analytics/option_chain_calibrator.hpp>
#include <ore/core/market_snapshot.hpp>
#include <ore/core/types.hpp>

namespace ore::analytics {

// =============================================================================
// Moneyness
// =============================================================================

/**
 * @brief Moneyness definitions supported by the smile builder.
 *
 * Each of these is a way to normalise strike so that "same moneyness"
 * means "same position on the smile" across different spots and terms.
 * Trade-offs:
 *
 *   * `Simple` = K/S. Intuitive (`1.20 = 20% above spot`) but not
 *     symmetric around ATM and not additive under log-returns.
 *   * `LogSimple` = ln(K/S). Symmetric around ATM (=0), additive, matches
 *     the log-normal frame Black-Scholes lives in. Standard in academic
 *     papers and the recommended default here.
 *   * `LogForward` = ln(K/F) with F = S*exp((r-q)T). Removes the
 *     cost-of-carry so that ATM-forward strikes map to 0 for every
 *     term — best for term-structure comparisons.
 */
enum class Moneyness {
    Simple,
    LogSimple,
    LogForward,
};

/** Human-readable name for CSV headers and logs. */
[[nodiscard]] constexpr std::string_view to_string(Moneyness m) noexcept {
    switch (m) {
        case Moneyness::Simple:      return "K/S";
        case Moneyness::LogSimple:   return "ln(K/S)";
        case Moneyness::LogForward:  return "ln(K/F)";
    }
    return "Unknown";
}

// =============================================================================
// VolatilitySmile
// =============================================================================

/**
 * @brief One volatility smile — calibrated IV as a function of strike
 *        for a single expiration.
 *
 * Vectors are strictly parallel: `strikes[i]`, `types[i]`, `moneyness[i]`,
 * `implied_volatility[i]` all describe the same calibrated contract.
 * Entries are sorted by ascending strike; ties (same strike, different
 * type — a call and a put) are ordered call-before-put.
 *
 * Only contracts with `was_calibrated() == true` are included.
 */
struct VolatilitySmile {
    /** Contract expiration date. */
    std::chrono::year_month_day expiration{};

    /** Time to expiration in years (ACT/365F). */
    double time_to_expiry{0.0};

    /** Strike prices, ascending. */
    std::vector<double> strikes{};

    /** Option type (Call/Put) parallel to `strikes`. */
    std::vector<ore::core::OptionType> types{};

    /** Moneyness values under `moneyness_convention`. */
    std::vector<double> moneyness{};

    /** Calibrated implied volatility (annualised, decimal). */
    std::vector<double> implied_volatility{};

    /** Which moneyness definition was used to populate `moneyness`. */
    Moneyness moneyness_convention{Moneyness::LogSimple};

    /** Number of points on the smile. */
    [[nodiscard]] std::size_t size() const noexcept { return strikes.size(); }
    /** True iff there are no calibrated points at this expiration. */
    [[nodiscard]] bool        empty() const noexcept { return strikes.empty(); }
};

/**
 * @brief Emit a *long-format* CSV with one row per smile point.
 *
 * Columns:
 *   `expiration, time_to_expiry, strike, option_type, moneyness_convention,
 *    moneyness, implied_volatility`.
 *
 * Multiple smiles can be concatenated into a single file by writing the
 * header once, then calling `write_smile_rows(...)` for each smile.
 * The `write_csv(std::span<...>)` overload below does exactly that.
 */
void write_csv(const VolatilitySmile& smile, std::ostream& os);

/** Write a header row plus one long-format row per point per smile. */
void write_csv(std::span<const VolatilitySmile> smiles, std::ostream& os);

/** File overload; opens `path` for writing. */
void write_csv(std::span<const VolatilitySmile> smiles,
               const std::filesystem::path& path);

// =============================================================================
// TermStructure
// =============================================================================

/**
 * @brief Term structure of at-the-money implied volatility.
 *
 * `expirations[i]`, `maturities[i]`, `atm_iv[i]` are parallel and
 * ascending in maturity. `atm_iv[i]` is linear-interpolated between the
 * two bracketing calibrated strikes at each expiration (see
 * `docs/VOLATILITY_ANALYTICS.md`). When only one calibrated point exists
 * or none brackets spot, the entry is left `NaN`.
 */
struct TermStructure {
    std::vector<std::chrono::year_month_day> expirations{};
    std::vector<double>                      maturities{};
    std::vector<double>                      atm_iv{};

    [[nodiscard]] std::size_t size() const noexcept { return maturities.size(); }
    [[nodiscard]] bool        empty() const noexcept { return maturities.empty(); }
};

void write_csv(const TermStructure& term, std::ostream& os);
void write_csv(const TermStructure& term, const std::filesystem::path& path);

// =============================================================================
// VolatilitySurface
// =============================================================================

/**
 * @brief Rectangular grid of implied vols across (expiration, strike).
 *
 * `implied_vols[i][j]` corresponds to `expirations[i]` and `strikes[j]`.
 * Missing entries (no calibrated contract at that pair) are `NaN`.
 * *Deliberately* not interpolated — the next milestone owns that step.
 * The classical single-value-per-cell convention is:
 *
 *   K <  spot -> put IV
 *   K >= spot -> call IV
 *   ties      -> the one closer to OTM (higher moneyness for calls, lower for puts)
 *
 * This is the same construction used to plot the "traditional" smile in
 * every options textbook (Hull Ch. 20, Rebonato & McKay).
 */
struct VolatilitySurface {
    std::vector<std::chrono::year_month_day> expirations{};
    std::vector<double>                      maturities{};
    std::vector<double>                      strikes{};
    std::vector<std::vector<double>>         implied_vols{};

    [[nodiscard]] std::size_t rows() const noexcept { return expirations.size(); }
    [[nodiscard]] std::size_t cols() const noexcept { return strikes.size(); }
    [[nodiscard]] bool        empty() const noexcept { return implied_vols.empty(); }
};

/**
 * @brief Emit the surface as long-format CSV.
 *
 * Columns: `expiration, time_to_expiry, strike, implied_volatility`.
 * Missing cells are written as empty fields (pandas reads them as `NaN`).
 * `df.pivot(index='expiration', columns='strike', values='implied_volatility')`
 * recovers the 2D grid.
 */
void write_csv(const VolatilitySurface& surface, std::ostream& os);
void write_csv(const VolatilitySurface& surface, const std::filesystem::path& path);

// =============================================================================
// SkewMetrics
// =============================================================================

/**
 * @brief Standard skew statistics for one expiration.
 *
 * Each field is `std::nullopt` when the chain does not cover enough
 * strikes to compute it (e.g. no 25-delta bracket, or no ATM bracket).
 */
struct SkewMetrics {
    std::chrono::year_month_day expiration{};
    double                      time_to_expiry{0.0};

    /** IV interpolated to K = spot (linear in strike). */
    std::optional<double> atm_iv;

    /** IV of the call whose BS delta equals `0.25`, linear interpolated
     *  in delta-space between the two calibrated calls bracketing that
     *  delta. See the .cpp for the exact formula. */
    std::optional<double> call_25delta_iv;

    /** Analog: put with BS delta = -0.25. */
    std::optional<double> put_25delta_iv;

    /** Risk reversal = `call_25delta_iv - put_25delta_iv`. Standard
     *  measure of asymmetric skew: negative in equity markets (puts
     *  more expensive than calls at the same |delta|). */
    std::optional<double> risk_reversal;

    /** Butterfly = `(call_25delta_iv + put_25delta_iv)/2 - atm_iv`.
     *  Measures wing convexity relative to ATM. Positive when the
     *  distribution is fatter-tailed than lognormal. */
    std::optional<double> butterfly;
};

void write_csv(std::span<const SkewMetrics> skews, std::ostream& os);
void write_csv(std::span<const SkewMetrics> skews, const std::filesystem::path& path);

// =============================================================================
// Statistics
// =============================================================================

/**
 * @brief Summary IV statistics over an arbitrary subset.
 *
 * Percentiles use numpy's linear-interpolation method:
 * `p-th percentile = sorted[floor(r)] + frac(r) * (sorted[ceil(r)] - sorted[floor(r)])`
 * with `r = (p/100)*(n-1)`. Guaranteed to lie in `[min, max]`.
 */
struct IVStatistics {
    std::size_t count{0};
    double min{0.0};
    double max{0.0};
    double mean{0.0};
    double median{0.0};
    /** Population standard deviation (n divisor, not n-1). Zero when
     *  count < 2. */
    double stddev{0.0};
    double p10{0.0};
    double p25{0.0};
    double p75{0.0};
    double p90{0.0};
};

/** Compute statistics over an arbitrary vector of IVs. Non-finite
 *  values are silently dropped. Empty input returns all-zeros. */
[[nodiscard]] IVStatistics compute_statistics(std::span<const double> values);

/** IV statistics over every calibrated contract in the report. */
[[nodiscard]] IVStatistics compute_statistics(const CalibrationReport& report);

/** IV statistics grouped by expiration date. Keys are ascending. */
[[nodiscard]] std::vector<std::pair<std::chrono::year_month_day, IVStatistics>>
compute_statistics_by_expiration(const CalibrationReport& report);

/** IV statistics for calls only and puts only, respectively. */
struct TypeStatistics {
    IVStatistics calls;
    IVStatistics puts;
};
[[nodiscard]] TypeStatistics compute_statistics_by_type(const CalibrationReport& report);

// =============================================================================
// Constructors
// =============================================================================

/**
 * @brief Group calibrated results by expiration and produce one smile
 *        per expiration.
 *
 * Return value ordered by ascending expiration; each smile has entries
 * ordered by ascending strike. Non-calibrated results are dropped.
 * `market.spot` and rates are used to compute moneyness (and, for
 * `LogForward`, the forward).
 */
[[nodiscard]] std::vector<VolatilitySmile> build_smiles(
    const CalibrationReport& report,
    const ore::core::MarketSnapshot& market,
    Moneyness convention = Moneyness::LogSimple);

/**
 * @brief Build the ATM-IV term structure.
 *
 * For each expiration, linearly interpolate IV to `K = market.spot`
 * using the two calibrated strikes that bracket spot. Expirations
 * without a bracket contribute `NaN`. Result is ascending in maturity.
 */
[[nodiscard]] TermStructure build_term_structure(
    const CalibrationReport& report,
    const ore::core::MarketSnapshot& market);

/**
 * @brief Build the (expiration, strike) IV grid.
 *
 * `strikes` is the sorted union of all strikes that appear at any
 * expiration. Cell `[i][j]` is the IV of the OTM contract at that pair
 * (put if `strikes[j] < spot`, else call), or `NaN` if that contract
 * was not calibrated at that expiration.
 */
[[nodiscard]] VolatilitySurface build_surface(
    const CalibrationReport& report,
    const ore::core::MarketSnapshot& market);

/**
 * @brief Compute standard skew metrics for every smile.
 *
 * `market` supplies the spot, rate, and dividend yield needed to
 * compute BS delta for each calibrated contract.
 */
[[nodiscard]] std::vector<SkewMetrics> compute_skew_metrics(
    std::span<const VolatilitySmile> smiles,
    const ore::core::MarketSnapshot& market);

} // namespace ore::analytics
