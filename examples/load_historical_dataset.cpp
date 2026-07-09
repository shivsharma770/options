/**
 * @file load_historical_dataset.cpp
 * @brief End-to-end demonstration of `HistoricalLoader`.
 *
 * Reads a ticker's historical option-chain tree from
 * `data/historical/raw/` (override with the first CLI argument) and
 * prints a compact summary:
 *
 *     Loaded:
 *       Ticker           : SPY
 *       Trading days     : 42
 *       Contracts loaded : 65 234
 *       Avg contracts/day: 1553.19
 *       Date range       : 2024-01-02 ... 2024-02-29
 *       Parse failures   : 0
 *       Missing dates    : 3 (first 3: 2024-01-15, 2024-02-19, ...)
 *
 * Usage
 * -----
 *
 *     ./example_load_historical_dataset [ticker] [root]
 *
 * Defaults: `ticker = SPY`, `root = data/historical/raw`.
 *
 * Runs in *lenient* mode so a single malformed day does not abort the
 * whole run; failures are printed after the summary. This is the
 * useful behaviour for a "how big is my dataset?" diagnostic.
 */

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include <ore/marketdata/historical_dataset.hpp>
#include <ore/marketdata/historical_loader.hpp>

int main(int argc, char** argv) try {
    const std::string ticker = (argc >= 2) ? argv[1] : "SPY";
    const std::filesystem::path root = (argc >= 3)
        ? std::filesystem::path{argv[2]}
        : std::filesystem::path{"data/historical/raw"};

    ore::marketdata::HistoricalLoader::Options opts{};
    opts.root   = root;
    opts.strict = false;   // Lenient: skip malformed days, keep going.

    const auto result = ore::marketdata::HistoricalLoader::load(ticker, opts);

    // Merge loader-only fields into the in-memory statistics so the
    // human printout carries everything in one block.
    auto stats = result.dataset.stats();
    stats.parse_failures = result.failed_days.size();
    stats.missing_dates  = result.missing_dates;

    std::cout << "Loaded:\n";
    stats.print(std::cout);

    if (!result.failed_days.empty()) {
        std::cout << "\nFailed days (" << result.failed_days.size() << "):\n";
        for (const auto& [d, msg] : result.failed_days) {
            std::cout << "  " << static_cast<int>(d.year()) << "-"
                      << static_cast<unsigned>(d.month()) << "-"
                      << static_cast<unsigned>(d.day())
                      << " : " << msg << "\n";
        }
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << "load_historical_dataset: " << e.what() << "\n";
    return 1;
}
