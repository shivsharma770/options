/**
 * @file monte_carlo_engine.hpp
 * @brief Risk-neutral Monte Carlo pricer for European vanilla options.
 *
 * ### Model
 *
 * Under the risk-neutral measure the underlying \f$S\f$ follows geometric
 * Brownian motion:
 * \f[
 *   dS/S = (r - q)\,dt + \sigma\,dW.
 * \f]
 * Its closed-form terminal distribution is
 * \f[
 *   S(T) \;=\; S(0)\,\exp\!\Bigl((r - q - \tfrac{1}{2}\sigma^2)\,T
 *                                 + \sigma \sqrt{T}\,Z\Bigr),\quad
 *   Z \sim \mathcal{N}(0, 1).
 * \f]
 * For European vanillas we need only \f$S(T)\f$, not the full path,
 * so this engine performs **one draw per path**. Multi-step path
 * simulation is a future milestone.
 *
 * ### Statistical estimator
 *
 * With \f$N\f$ i.i.d. samples \f$\{S_i(T)\}\f$ the price estimator is
 * \f[
 *   \hat{V} \;=\; e^{-rT}\,\frac{1}{N}\sum_{i=1}^{N} \text{payoff}(S_i(T)),
 * \f]
 * an unbiased estimator whose standard error converges as
 * \f$O(1/\sqrt{N})\f$. The engine computes running mean and variance
 * with **Welford's online algorithm** (see the .cpp), so path payoffs
 * are never stored.
 *
 * ### Variance reduction
 *
 * When `Config::antithetic_variates` is enabled, each Welford sample
 * is the average of two payoffs — one at \f$Z\f$ and one at \f$-Z\f$.
 * For monotone payoffs the two are negatively correlated, so their
 * average has strictly lower variance than either alone.
 *
 * ### Greeks
 *
 * Optional (`Config::compute_greeks`, off by default). Bump-and-revalue
 * using the **same RNG seed** for every bumped run — a form of common
 * random numbers that cancels the noise between \f$V(x + h)\f$ and
 * \f$V(x - h)\f$ so the finite difference sees only the derivative
 * signal, not the ± ε per-sample noise.
 *
 * ### Reproducibility
 *
 * Fixed `Config::seed` → bit-identical prices across runs on the same
 * standard-library implementation. The `std::normal_distribution`
 * transform is implementation-defined across libc++/libstdc++/MSVC,
 * so identical prices are guaranteed only within one build environment.
 *
 * ### References
 *
 * - Glasserman, P. (2003). *Monte Carlo Methods in Financial
 *   Engineering*. Springer. Chapters 1, 2, 4.
 * - Welford, B.P. (1962). "Note on a method for calculating corrected
 *   sums of squares and products". Technometrics 4(3).
 * - Kahan, W. (1965). "Further remarks on reducing truncation errors".
 *   Comm. ACM 8(1).
 * - Boyle, P. (1977). "Options: A Monte Carlo Approach". JFE 4(3).
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include <ore/core/market_snapshot.hpp>
#include <ore/core/option.hpp>
#include <ore/core/types.hpp>
#include <ore/pricing/pricing_engine.hpp>
#include <ore/pricing/pricing_result.hpp>

namespace ore::pricing {

/**
 * @class MonteCarloEngine
 * @brief Direct-terminal Monte Carlo pricer for European vanilla options.
 *
 * Interchangeable with `BlackScholesEngine` and `BinomialTreeEngine`:
 * same abstract interface, same `PricingResult` shape. Population of
 * `standard_error` and `confidence_interval_95` distinguishes it from
 * the deterministic engines.
 *
 * Stateless with respect to market data. A single engine instance can
 * be reused across many pricing calls (each call constructs its own
 * generator seeded from `Config::seed`).
 */
class MonteCarloEngine final : public PricingEngine {
public:
    /**
     * @brief Engine-wide configuration.
     */
    struct Config {
        /** Number of Welford samples. In non-antithetic mode this is
         *  the total number of RNG draws and payoff evaluations. In
         *  antithetic mode it is the number of *pairs* — each pair
         *  averages payoffs at Z and −Z. Larger N → lower standard
         *  error (\f$O(1/\sqrt{N})\f$) at linear cost. */
        std::size_t paths{1'000'000};

        /** Seed for the internal RNG. Two calls with the same seed
         *  produce bit-identical prices on the same platform. */
        std::uint64_t seed{42};

        /** If true, each Welford sample averages the payoff at Z and
         *  −Z, halving the variance for monotone payoffs at the cost
         *  of one extra `exp` per pair (RNG cost is unchanged: −Z is
         *  free once Z is drawn). Recommended for vanilla options. */
        bool antithetic_variates{true};

        /** If true, populate `PricingResult::greeks` via bump-and-
         *  revalue. This costs ~8× additional runs of the same size;
         *  disable for benchmarking pure pricing throughput. Greeks
         *  use common random numbers (same seed) across bumps, which
         *  reduces variance by 2-3 orders of magnitude compared to
         *  naive bump-and-revalue. */
        bool compute_greeks{false};
    };

    /**
     * @brief Raw scalar inputs. Same fields as the other engines'
     *        `Inputs` structs (minus `ExerciseStyle`, since only
     *        European exercise is supported).
     */
    struct Inputs {
        double spot{0.0};              ///< S, > 0.
        double strike{0.0};            ///< K, > 0.
        double rate{0.0};              ///< r, continuously compounded (decimal).
        double dividend_yield{0.0};    ///< q, continuously compounded (decimal).
        double volatility{0.0};        ///< sigma, annualised (decimal). >= 0.
        double time_to_expiry{0.0};    ///< T in years (ACT/365F). >= 0.
        ore::core::OptionType type{ore::core::OptionType::Call};
    };

    /** Default construction: 1M paths, seed 42, antithetic on, Greeks off. */
    MonteCarloEngine();

    /** Construct with a specific configuration. Rebuilds the engine
     *  name string; see `name()`. */
    explicit MonteCarloEngine(Config config);

    /** Access the current engine configuration (immutable). */
    [[nodiscard]] const Config& config() const noexcept { return config_; }

    /**
     * @copydoc PricingEngine::price
     *
     * Derives \f$T\f$ from `market.valuation_date` and
     * `option.expiration` (ACT/365F). Rejects American exercise with
     * `std::invalid_argument` — this milestone supports European only.
     */
    [[nodiscard]] PricingResult price(
        const ore::core::Option& option,
        const ore::core::MarketSnapshot& market) const override;

    /**
     * @brief Price directly from raw scalar inputs.
     *
     * @throws std::invalid_argument on any precondition violation
     *         (see `Inputs` and `Config`).
     *
     * @return A fully-populated `PricingResult`:
     *         - `price`      : discounted risk-neutral sample mean.
     *         - `greeks`     : if `config.compute_greeks == true`, else zero.
     *         - `engine_name`: e.g. `"MonteCarlo(paths=1000000, seed=42, antithetic=true)"`.
     *         - `iterations` : total number of Welford samples processed.
     *         - `standard_error`         : \f$\sigma_{\text{sample}} / \sqrt{N}\f$.
     *         - `confidence_interval_95` : `(price - 1.96·SE, price + 1.96·SE)`.
     */
    [[nodiscard]] PricingResult price(const Inputs& inputs) const;

    /** @copydoc PricingEngine::name */
    [[nodiscard]] std::string_view name() const noexcept override {
        return name_;
    }

    /**
     * @brief Standard-normal quantile used to convert `standard_error`
     *        to the 95% confidence interval half-width.
     *
     * Exposed as a public constant so tests and consumers can produce
     * consistent CIs from a bare `standard_error` when they need to.
     */
    static constexpr double kZ95 = 1.959963984540054;

private:
    Config      config_{};
    std::string name_{};
};

} // namespace ore::pricing
