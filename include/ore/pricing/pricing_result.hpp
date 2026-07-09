/**
 * @file pricing_result.hpp
 * @brief Structured output of a pricing engine.
 *
 * Every concrete pricing engine (Black-Scholes closed form, binomial tree,
 * Monte Carlo, finite-difference, ...) returns this same struct. The
 * shape is deliberately chosen so the interface stays stable as new
 * engines land:
 *
 * - **`price`** and **`greeks`** are always present. Every engine can
 *   compute them (Greeks via analytic derivative, tree finite-differences,
 *   pathwise, or bump-and-revalue). Engines that intentionally skip
 *   Greeks leave the struct default-constructed (all zeros) and document
 *   that in their public API.
 *
 * - **Diagnostics** — `engine_name`, `iterations`, `standard_error` —
 *   describe how the price was produced. They are `std::optional` (or an
 *   empty string) so that a value of "0" is unambiguous: engines that
 *   don't have iterations to report leave `iterations` empty; engines
 *   that finished after literally zero iterations set it to `0`.
 *
 * ### Engine-by-engine expected shape
 *
 * | Engine          | iterations         | standard_error        |
 * |-----------------|--------------------|-----------------------|
 * | Black-Scholes   | `std::nullopt`     | `std::nullopt`        |
 * | Binomial trees  | number of steps    | `std::nullopt`        |
 * | Monte Carlo     | number of paths    | sample SE of price    |
 * | Finite diff.    | number of grid pts | `std::nullopt`        |
 *
 * Extra diagnostic fields (bias estimate, seed used, wall-clock elapsed,
 * ...) can be added to this struct without breaking existing call sites,
 * because every consumer only reads the fields it cares about.
 */
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include <ore/pricing/greeks.hpp>

namespace ore::pricing {

/**
 * @brief Result of a single pricing evaluation.
 */
struct PricingResult {
    /** Fair value in the underlying's currency. */
    double price{0.0};

    /** First-order sensitivities. Left default-constructed (zeros) by
     *  engines that do not compute Greeks; see engine documentation. */
    Greeks greeks{};

    /** Short human-readable engine identifier, e.g. `"BlackScholes"`,
     *  `"Binomial(N=200)"`, `"MonteCarlo(paths=50000, seed=42)"`. Empty
     *  string if not set. Meant for logging, model-comparison tables, and
     *  regression fixtures — not a stable machine-readable API. */
    std::string engine_name{};

    /** Iterations / paths / tree steps consumed, when the engine has a
     *  meaningful count. Left empty by closed-form engines (Black-Scholes),
     *  set by iterative or stochastic engines. */
    std::optional<std::size_t> iterations{};

    /** Estimated standard error of `price`, when the engine is stochastic
     *  and can produce one (Monte Carlo, quasi-MC with control variates,
     *  ...). Left empty by deterministic engines. */
    std::optional<double> standard_error{};

    /** 95% confidence interval for `price`, `(lower, upper)`. Set only
     *  by stochastic engines that also populate `standard_error`. The
     *  95% level is the trader-desk default; consumers wanting any
     *  other level can derive it from `standard_error` directly
     *  (upper = price ± z(1 - alpha/2) * standard_error). */
    std::optional<std::pair<double, double>> confidence_interval_95{};

    // Deliberately no `operator==`: `price`, `greeks`, `standard_error`,
    // and `confidence_interval_95` are floating-point; equality
    // comparisons are a footgun. Tests should compare fields individually
    // with `approximately_equal`.
};

} // namespace ore::pricing
