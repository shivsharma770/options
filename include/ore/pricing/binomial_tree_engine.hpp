/**
 * @file binomial_tree_engine.hpp
 * @brief Cox-Ross-Rubinstein (CRR) binomial-tree pricer.
 *
 * ### Model
 *
 * Time-to-expiry \f$T\f$ is discretised into \f$N\f$ steps of size
 * \f$\Delta t = T/N\f$. At each node the underlying moves up by factor
 * \f$u\f$ or down by factor \f$d\f$ with risk-neutral probabilities
 * \f$p\f$ and \f$1-p\f$ respectively:
 * \f{align*}{
 *   u &= e^{\sigma \sqrt{\Delta t}} \\
 *   d &= 1/u \\
 *   p &= \frac{e^{(r-q)\Delta t} - d}{u - d}
 * \f}
 * The tree is *recombining* (\f$u \cdot d = 1\f$), so an \f$N\f$-step
 * lattice has \f$N+1\f$ terminal nodes and \f$(N+1)(N+2)/2\f$ total
 * nodes rather than \f$2^N\f$.
 *
 * The price at each node is the risk-neutral discounted expectation of
 * its two children; for American exercise, this is compared against the
 * intrinsic value and the max is taken. Backward induction from
 * terminal payoff to \f$t = 0\f$ produces the fair value.
 *
 * ### Why CRR?
 *
 * Alternatives (Jarrow-Rudd, Tian) match different moments of the
 * lognormal distribution. CRR is the pedagogical standard: log-symmetric
 * (u·d = 1), reduces to a single parameter \f$u\f$, and its
 * price-oscillation convergence to Black-Scholes is well studied. See
 * `docs/BINOMIAL_TREE.md` for the derivation and the trade-offs.
 *
 * ### Exercise style
 *
 * Both European and American are supported. Exercise style lives on the
 * `Inputs` struct (mirroring `Option::exercise`) rather than on
 * `Config`: the exercise flag describes the *contract*, not the engine.
 * A single `BinomialTreeEngine` can price both flavours interchangeably.
 *
 * ### Complexity
 *
 * *Time* O(N²), *memory* O(N). See the .cpp for how the O(N) memory
 * bound is achieved via a rolling vector.
 *
 * ### Greeks
 *
 * All five Greeks are computed via **bump-and-revalue** using the same
 * engine. This adds ~8× tree evaluations per call; disable via
 * `Config::compute_greeks = false` when benchmarking pure pricing
 * speed. Bump sizes and central-vs-forward-difference choices are
 * documented alongside the implementation.
 *
 * ### References
 * - Cox, J.C., Ross, S.A., and Rubinstein, M. (1979). *Option Pricing:
 *   A Simplified Approach*. Journal of Financial Economics 7.
 * - Hull, J. (2018). *Options, Futures, and Other Derivatives*, 10th Ed.,
 *   Chapter 21.
 * - Broadie, M. and Detemple, J. (1996). *American Option Valuation:
 *   New Bounds, Approximations, and a Comparison of Existing Methods*.
 *   Review of Financial Studies 9(4).
 */
#pragma once

#include <cstddef>
#include <string>
#include <string_view>

#include <ore/core/market_snapshot.hpp>
#include <ore/core/option.hpp>
#include <ore/core/types.hpp>
#include <ore/pricing/pricing_engine.hpp>
#include <ore/pricing/pricing_result.hpp>

namespace ore::pricing {

/**
 * @class BinomialTreeEngine
 * @brief Cox-Ross-Rubinstein tree pricer for European and American vanilla options.
 *
 * Interchangeable with `BlackScholesEngine` — both implement the same
 * `PricingEngine` interface and populate the same `PricingResult`.
 * A single instance is stateless with respect to market data and safe
 * to reuse across many pricing calls.
 */
class BinomialTreeEngine final : public PricingEngine {
public:
    /**
     * @brief Engine-wide configuration.
     *
     * @note The specification for this milestone suggested `bool american`
     *       as a Config field. We instead put `ExerciseStyle exercise` on
     *       the `Inputs` struct, mirroring `Option::exercise`. Exercise
     *       style is a property of the *contract*, not the engine; keeping
     *       it out of Config lets one engine instance price European and
     *       American contracts interchangeably.
     */
    struct Config {
        /** Number of time steps \f$N\f$. Larger N → more accurate but
         *  O(N²) time. Practical range: 50–5000. */
        std::size_t steps{500};

        /** If false, Greeks in `PricingResult` are left zero.
         *  Skips ~8 additional tree evaluations per call — useful for
         *  benchmarks and convergence studies that only need price. */
        bool compute_greeks{true};
    };

    /**
     * @brief Raw scalar inputs. Exposed for callers who do not want to
     *        build a full `Option` + `MarketSnapshot` pair (research
     *        notebooks, calibrators, test fixtures).
     *
     * `time_to_expiry` uses the same ACT/365F convention as
     * `MarketSnapshot`. Preconditions (all validated at call time):
     *
     *   - `spot > 0`, `strike > 0`, all fields finite
     *   - `volatility >= 0`, `time_to_expiry >= 0`
     *   - `rate` and `dividend_yield` unconstrained (negative rates OK)
     */
    struct Inputs {
        double spot{0.0};
        double strike{0.0};
        double rate{0.0};
        double dividend_yield{0.0};
        double volatility{0.0};
        double time_to_expiry{0.0};
        ore::core::OptionType     type{ore::core::OptionType::Call};
        ore::core::ExerciseStyle  exercise{ore::core::ExerciseStyle::European};
    };

    /** Default construction: 500 steps, Greeks enabled. */
    BinomialTreeEngine();

    /** Construct with a specific configuration. Rebuilds the engine
     *  name string; use `name()` to see it. */
    explicit BinomialTreeEngine(Config config);

    /** Access the current engine configuration (immutable). */
    [[nodiscard]] const Config& config() const noexcept { return config_; }

    /**
     * @copydoc PricingEngine::price
     *
     * Reads `option.exercise` to decide between European and American
     * pricing. Delegates to the `Inputs`-overload after deriving `T`
     * from `market.valuation_date` and `option.expiration` (ACT/365F).
     */
    [[nodiscard]] PricingResult price(
        const ore::core::Option& option,
        const ore::core::MarketSnapshot& market) const override;

    /**
     * @brief Price directly from raw scalar inputs.
     *
     * @throws std::invalid_argument  If preconditions on `Inputs` are
     *         violated (see the struct docs) or if `Config::steps == 0`.
     *
     * @return A fully-populated `PricingResult`:
     *         - `price`: the CRR tree value.
     *         - `greeks`: bump-and-revalue estimates (or all-zero if
     *           `Config::compute_greeks == false`).
     *         - `engine_name`: e.g. `"Binomial(CRR, N=500)"`.
     *         - `iterations`: the step count `N`.
     *         - `standard_error`: `std::nullopt` (tree is deterministic).
     */
    [[nodiscard]] PricingResult price(const Inputs& inputs) const;

    /** Engine identifier including the step count, e.g.
     *  `"Binomial(CRR, N=500)"`. Backed by an owned string_view-compatible
     *  member; safe to use for the lifetime of `*this`. */
    [[nodiscard]] std::string_view name() const noexcept override {
        return name_;
    }

private:
    Config      config_{};
    std::string name_{};
};

} // namespace ore::pricing
