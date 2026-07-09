/**
 * @file benchmark_all_engines.cpp
 * @brief End-to-end driver for the `ore::benchmark` module.
 *
 * Runs three flavours of benchmark and writes each to its own CSV:
 *
 *   1. The **standard 12-case suite** across Black-Scholes,
 *      Binomial (N=500, Greeks on) and Monte Carlo (P=200 000, seed=42,
 *      antithetic on, Greeks on). Output: `benchmark_suite.csv`.
 *
 *   2. A **Binomial convergence sweep** on the ATM call over
 *      { 10, 25, 50, 100, 250, 500, 1000, 2500, 5000 } steps.
 *      Output: `benchmark_binomial_convergence.csv`.
 *
 *   3. A **Monte Carlo convergence sweep** on the same case over
 *      { 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6 } paths, shared
 *      seed 42, antithetic variates on.
 *      Output: `benchmark_monte_carlo_convergence.csv`.
 *
 * Also dumps a compact human-readable summary of the standard-suite
 * report to stdout so the executable is useful even without the CSV
 * files.
 *
 * CSVs are written to the current working directory. Use with the
 * Python plotters in `python/` to produce publication figures.
 *
 * Exit code: 0 on success, 1 on any thrown exception.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <ios>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <ore/benchmark/benchmark.hpp>
#include <ore/pricing/binomial_tree_engine.hpp>
#include <ore/pricing/black_scholes_engine.hpp>
#include <ore/pricing/monte_carlo_engine.hpp>
#include <ore/pricing/pricing_engine.hpp>

namespace {

using ore::benchmark::BenchmarkReport;
using ore::benchmark::BenchmarkRunner;
using ore::benchmark::standard_benchmark_suite;
using ore::pricing::BinomialTreeEngine;
using ore::pricing::BlackScholesEngine;
using ore::pricing::MonteCarloEngine;
using ore::pricing::PricingEngine;

// Build the three-engine mix used for the standard suite. Greeks are
// enabled on both non-analytical engines so that the report includes
// Delta/Gamma/etc. for the Python `plot_greeks.py` script.
std::vector<std::unique_ptr<PricingEngine>> make_suite_engines() {
    std::vector<std::unique_ptr<PricingEngine>> engines;
    engines.push_back(std::make_unique<BlackScholesEngine>());

    BinomialTreeEngine::Config bt_cfg{};
    bt_cfg.steps = 500;
    bt_cfg.compute_greeks = true;
    engines.push_back(std::make_unique<BinomialTreeEngine>(bt_cfg));

    MonteCarloEngine::Config mc_cfg{};
    mc_cfg.paths = 200'000;
    mc_cfg.seed = 42;
    mc_cfg.antithetic_variates = true;
    mc_cfg.compute_greeks = true;
    engines.push_back(std::make_unique<MonteCarloEngine>(mc_cfg));

    return engines;
}

// Pretty-print a compact per-case summary table to stdout. The full
// per-Greek and per-engine data lives in the CSV.
void print_summary(const BenchmarkReport& report) {
    std::cout << std::left
              << std::setw(24) << "case"
              << std::setw(28) << "engine"
              << std::right
              << std::setw(14) << "price"
              << std::setw(14) << "abs_error"
              << std::setw(14) << "runtime(us)"
              << std::setw(10) << "iters"
              << "\n";
    std::cout << std::string(104, '-') << "\n";

    std::cout << std::fixed;
    for (const auto& r : report.rows) {
        std::cout << std::left
                  << std::setw(24) << r.case_name
                  << std::setw(28) << r.engine_name
                  << std::right
                  << std::setw(14) << std::setprecision(6) << r.price;
        if (r.absolute_error.has_value()) {
            std::cout << std::setw(14) << std::setprecision(6) << *r.absolute_error;
        } else {
            std::cout << std::setw(14) << "-";
        }
        std::cout << std::setw(14) << std::setprecision(2) << r.runtime_us;
        if (r.iterations.has_value()) {
            std::cout << std::setw(10) << *r.iterations;
        } else {
            std::cout << std::setw(10) << "-";
        }
        std::cout << "\n";
    }
}

} // namespace

int main() try {
    const std::filesystem::path suite_csv = "benchmark_suite.csv";
    const std::filesystem::path binomial_csv = "benchmark_binomial_convergence.csv";
    const std::filesystem::path mc_csv = "benchmark_monte_carlo_convergence.csv";

    BenchmarkRunner runner{};  // Default config: 3 median reps, "BlackScholes" ref.

    // --- Standard 12-case suite ---
    auto engines = make_suite_engines();
    const auto suite_report = runner.run(engines, standard_benchmark_suite());
    suite_report.write_csv(suite_csv);
    print_summary(suite_report);
    std::cout << "\nWrote " << suite_csv << " (" << suite_report.rows.size()
              << " rows)\n";

    // --- Binomial convergence sweep on the ATM call ---
    const auto atm_call = standard_benchmark_suite().front();
    const std::vector<std::size_t> step_counts = {
        10, 25, 50, 100, 250, 500, 1000, 2500, 5000
    };
    const auto bt_report = runner.run_binomial_convergence(atm_call, step_counts);
    bt_report.write_csv(binomial_csv);
    std::cout << "Wrote " << binomial_csv << " ("
              << bt_report.rows.size() << " rows)\n";

    // --- Monte Carlo convergence sweep on the same case ---
    const std::vector<std::size_t> path_counts = {
        1'000, 5'000, 10'000, 50'000, 100'000, 500'000, 1'000'000, 5'000'000
    };
    const auto mc_report = runner.run_monte_carlo_convergence(
        atm_call, path_counts, /*seed=*/42, /*antithetic=*/true);
    mc_report.write_csv(mc_csv);
    std::cout << "Wrote " << mc_csv << " ("
              << mc_report.rows.size() << " rows)\n";

    return 0;
} catch (const std::exception& e) {
    std::cerr << "benchmark_all_engines: " << e.what() << "\n";
    return 1;
}
