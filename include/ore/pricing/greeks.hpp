/**
 * @file greeks.hpp
 * @brief The five first-order sensitivities every pricing engine reports.
 *
 * `Greeks` is a plain aggregate — no invariants, no invisible state. Every
 * pricing engine (Black-Scholes closed form, binomial tree, Monte Carlo,
 * finite difference, ...) populates it in the same units, so callers can
 * compare across engines and diff results without translating conventions.
 *
 * ### Unit convention
 * All fields are in **"natural" units**: partial derivatives with respect
 * to their underlying variable, evaluated per unit change of that
 * variable.
 *
 *  - `delta`  is `dV / dS`.
 *  - `gamma`  is `d^2 V / dS^2`.
 *  - `theta`  is `dV / dt`, **per year** (calendar-year fraction from
 *              valuation date to expiration). To render "theta per calendar
 *              day", multiply by `1.0 / 365.0` at display time. This engine
 *              layer does not do that scaling.
 *  - `vega`   is `dV / dsigma`, **per unit of volatility** (i.e. per 1.0
 *              = 100 vol points). To render "vega per 1 vol point" (the
 *              trader-facing convention) multiply by `0.01` at display.
 *  - `rho`    is `dV / dr`, **per unit of rate** (i.e. per 1.0 = 100%).
 *              For "rho per basis point" multiply by `1.0e-4` at display.
 *
 * Presentation-layer scaling (per-day theta, per-1%-vol vega, per-bp rho)
 * is deliberately kept out of the core pricing library so every model
 * speaks the same language.
 *
 * ### Missing values
 * Engines that do not compute Greeks are expected to leave the struct
 * default-constructed (all zeros). Engines that compute some but not
 * others should document which they populate. This is a **convention**,
 * not enforced by the type — see `PricingResult` for the composed shape.
 */
#pragma once

namespace ore::pricing {

/**
 * @brief First-order option sensitivities returned by every pricing engine.
 */
struct Greeks {
    /** \f$ \Delta = \frac{\partial V}{\partial S} \f$ — price sensitivity to a
     *  unit change in the underlying spot. Unitless. */
    double delta{0.0};

    /** \f$ \Gamma = \frac{\partial^2 V}{\partial S^2} \f$ — rate of change of
     *  delta with respect to spot. Non-negative for standard options. */
    double gamma{0.0};

    /** \f$ \Theta = \frac{\partial V}{\partial t} \f$ — sensitivity to the
     *  passage of calendar time, **per year**. Typically negative for long
     *  option positions (options decay). */
    double theta{0.0};

    /** \f$ \mathcal{V} = \frac{\partial V}{\partial \sigma} \f$ — sensitivity
     *  to volatility, **per unit of vol** (i.e. per 1.0 = 100 vol points).
     *  Non-negative for standard options. */
    double vega{0.0};

    /** \f$ \rho = \frac{\partial V}{\partial r} \f$ — sensitivity to the
     *  risk-free rate, **per unit of rate** (i.e. per 1.0 = 100%). */
    double rho{0.0};

    /** Structural equality. Fine for tests that construct expected Greeks
     *  from bit-identical literals; use `approximately_equal` for
     *  numerical comparisons. */
    friend bool operator==(const Greeks&, const Greeks&) = default;
};

} // namespace ore::pricing
