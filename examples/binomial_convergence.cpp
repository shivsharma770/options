/**
 * @file binomial_convergence.cpp
 * @brief Convergence + timing benchmark for `BinomialTreeEngine`.
 *
 * Prints a table:
 *
 *   Steps | Tree Price | BS Price | Abs Error | Runtime (μs)
 *
 * for a canonical European call with S = K = 100, r = 5 %, sigma = 20 %,
 * q = 0, T = 1 year. The step counts follow the sequence requested in
 * the milestone spec:
 *
 *   10, 25, 50, 100, 250, 500, 1000, 2500, 5000.
 *
 * The Black-Scholes price is computed once and reused as the reference.
 * Runtime is the median of a small number of repetitions to smooth over
 * scheduling noise (rerunning tiny trees ~1000 times is essentially
 * free).
 *
 * The output is intended to be diffed across changes and, if useful,
 * imported into a plotting notebook. Exit code is always 0 unless a
 * pricing call throws (in which case the exception is caught, logged,
 * and a non-zero code is returned).
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>
#include <vector>

#include <ore/core/types.hpp>
#include <ore/pricing/binomial_tree_engine.hpp>
#include <ore/pricing/black_scholes_engine.hpp>

namespace {

using ore::core::ExerciseStyle;
using ore::core::OptionType;
using ore::pricing::BinomialTreeEngine;
using ore::pricing::BlackScholesEngine;

BinomialTreeEngine::Inputs canonical_case() {
    return {
        .spot           = 100.0,
        .strike         = 100.0,
        .rate           = 0.05,
        .dividend_yield = 0.0,
        .volatility     = 0.20,
        .time_to_expiry = 1.0,
        .type           = OptionType::Call,
        .exercise       = ExerciseStyle::European,
    };
}

BlackScholesEngine::Inputs to_bs(const BinomialTreeEngine::Inputs& in) {
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

/**
 * Time one tree evaluation `reps` times and return the median wall-clock
 * duration in microseconds. Greeks are disabled so this benchmark
 * measures pure pricing throughput.
 */
double median_us(std::size_t steps,
                 const BinomialTreeEngine::Inputs& inputs,
                 std::size_t reps)
{
    BinomialTreeEngine engine({.steps = steps, .compute_greeks = false});
    std::vector<double> samples;
    samples.reserve(reps);
    for (std::size_t i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto r  = engine.price(inputs);
        const auto t1 = std::chrono::steady_clock::now();
        // Prevent the optimiser from eliding the call.
        volatile double sink = r.price;
        (void)sink;
        samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::nth_element(samples.begin(),
                     samples.begin() + static_cast<std::ptrdiff_t>(samples.size() / 2),
                     samples.end());
    return samples[samples.size() / 2];
}

/**
 * Pick a sensible number of repetitions given the step count. Small
 * trees need many repetitions to average out timer noise; very large
 * trees need fewer since each call is already >> the timer resolution.
 */
constexpr std::size_t reps_for(std::size_t steps) {
    if (steps <=   50) return 5000;
    if (steps <=  250) return 1000;
    if (steps <= 1000) return  100;
    return 20;
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
                  << std::fixed << std::setprecision(10) << bs_price << "\n\n";

        std::cout << std::left
                  << std::setw(8)  << "Steps"
                  << std::setw(18) << "Tree Price"
                  << std::setw(18) << "BS Price"
                  << std::setw(18) << "Abs Error"
                  << std::setw(14) << "Runtime (us)" << "\n";
        std::cout << std::string(76, '-') << "\n";

        for (std::size_t steps : {10U, 25U, 50U, 100U, 250U, 500U,
                                  1000U, 2500U, 5000U})
        {
            BinomialTreeEngine engine({.steps = steps, .compute_greeks = false});
            const double tree = engine.price(inputs).price;
            const double err  = std::abs(tree - bs_price);
            const double us   = median_us(steps, inputs, reps_for(steps));

            std::cout << std::left
                      << std::setw(8)  << steps
                      << std::fixed << std::setprecision(10)
                      << std::setw(18) << tree
                      << std::setw(18) << bs_price
                      << std::scientific << std::setprecision(3)
                      << std::setw(18) << err
                      << std::defaultfloat << std::setprecision(3)
                      << std::setw(14) << us
                      << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "benchmark failed: " << e.what() << "\n";
        return 1;
    }
}
