/**
 * @file monte_carlo_convergence.cpp
 * @brief Convergence + timing benchmark for `MonteCarloEngine`.
 *
 * Prints a table:
 *
 *   Paths | Price | BS Error | Std Error | 95 % CI | Runtime (ms)
 *
 * for a canonical European ATM call:
 *   S = K = 100, r = 5 %, q = 0, sigma = 20 %, T = 1 year.
 *
 * Path counts follow the milestone spec:
 *   1,000  5,000  10,000  50,000  100,000  500,000  1,000,000  5,000,000
 *
 * Antithetic variates are on (default). Greeks are off — this benchmark
 * measures pricing throughput and convergence, not Greek precision.
 * The Black-Scholes price is computed once and reused as the reference.
 *
 * Exit code is 0 on success, 1 on any thrown exception.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>
#include <vector>

#include <ore/core/types.hpp>
#include <ore/pricing/black_scholes_engine.hpp>
#include <ore/pricing/monte_carlo_engine.hpp>

namespace {

using ore::core::OptionType;
using ore::pricing::BlackScholesEngine;
using ore::pricing::MonteCarloEngine;

MonteCarloEngine::Inputs canonical_case() {
    return {
        .spot           = 100.0,
        .strike         = 100.0,
        .rate           = 0.05,
        .dividend_yield = 0.0,
        .volatility     = 0.20,
        .time_to_expiry = 1.0,
        .type           = OptionType::Call,
    };
}

BlackScholesEngine::Inputs to_bs(const MonteCarloEngine::Inputs& in) {
    return {
        .spot           = in.spot,
        .strike         = in.strike,
        .rate           = in.rate,
        .dividend_yield = in.dividend_yield,
        .volatility     = in.volatility,
        .time_to_expiry = in.time_to_expiry,
        .type           = in.type,
    };
}

}  // namespace

int main() {
    try {
        const auto inputs = canonical_case();

        BlackScholesEngine bs;
        auto bs_in = to_bs(inputs);
        const double bs_price = bs.price(bs_in).price;

        std::cout << "European ATM call: S=K=100, r=5%, q=0, sigma=20%, T=1y\n";
        std::cout << "Black-Scholes reference price: "
                  << std::fixed << std::setprecision(10) << bs_price << "\n";
        std::cout << "Antithetic variates: on   |   Seed: 42\n\n";

        std::cout << std::left
                  << std::setw(12) << "Paths"
                  << std::setw(16) << "Price"
                  << std::setw(14) << "BS Error"
                  << std::setw(14) << "Std Error"
                  << std::setw(28) << "95% CI"
                  << std::setw(14) << "Runtime (ms)" << "\n";
        std::cout << std::string(98, '-') << "\n";

        for (std::size_t paths : {1'000U, 5'000U, 10'000U, 50'000U, 100'000U,
                                  500'000U, 1'000'000U, 5'000'000U})
        {
            MonteCarloEngine engine({
                .paths = paths, .seed = 42,
                .antithetic_variates = true,
                .compute_greeks = false,
            });

            const auto t0 = std::chrono::steady_clock::now();
            const auto r = engine.price(inputs);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();

            const double err = std::abs(r.price - bs_price);
            const double se  = r.standard_error.value_or(0.0);

            char ci_buf[48];
            if (r.confidence_interval_95.has_value()) {
                const auto [lo, hi] = *r.confidence_interval_95;
                std::snprintf(ci_buf, sizeof(ci_buf),
                              "[%.6f, %.6f]", lo, hi);
            } else {
                std::snprintf(ci_buf, sizeof(ci_buf), "n/a");
            }

            std::cout << std::left
                      << std::setw(12) << paths
                      << std::fixed << std::setprecision(6)
                      << std::setw(16) << r.price
                      << std::scientific << std::setprecision(3)
                      << std::setw(14) << err
                      << std::setw(14) << se
                      << std::setw(28) << ci_buf
                      << std::defaultfloat << std::setprecision(3)
                      << std::setw(14) << ms
                      << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "benchmark failed: " << e.what() << "\n";
        return 1;
    }
}
