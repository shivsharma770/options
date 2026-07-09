/**
 * @file historical_calibration.cpp
 * @brief End-to-end demonstration of `HistoricalCalibrationStudy`.
 *
 * Loads the SPY historical archive from `data/historical/spy/`,
 * runs `HistoricalCalibrationStudy` across every trading day, and
 * writes five CSVs to `data/generated/research/`:
 *
 *   - `calibration.csv`    — per-contract IV + solver diagnostics
 *   - `smiles.csv`         — long-format IV smiles per day
 *   - `term_structure.csv` — ATM IV vs expiration per day
 *   - `surface.csv`        — long-format IV grid per day
 *   - `skew.csv`           — 25-delta skew metrics per day
 *
 * These are exactly the CSVs the existing `python/plot_smile.py`,
 * `plot_surface.py`, and `plot_term_structure.py` scripts consume,
 * except now the time-series dimension is baked in via the `date`
 * column.
 *
 * Usage
 * -----
 *
 *     ./example_historical_calibration [root] [ticker] [threads] [max_days]
 *
 * Defaults
 *   root     = data/historical/spy
 *   ticker   = SPY
 *   threads  = std::thread::hardware_concurrency()
 *   max_days = 0 (all days on disk)
 */

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <ore/marketdata/historical_dataset.hpp>
#include <ore/marketdata/historical_loader.hpp>
#include <ore/marketdata/historical_snapshot.hpp>
#include <ore/research/historical_calibration_study.hpp>
#include <ore/research/historical_research_engine.hpp>
#include <ore/research/research_report.hpp>

namespace {

/// Simple stderr progress bar identical in spirit to the one in
/// `historical_research.cpp`. Rewrites the same line on every
/// percentage change.
void render_progress(std::size_t done, std::size_t total) {
    static int last_pct = -1;
    if (total == 0) return;
    const int pct = static_cast<int>((100.0 * static_cast<double>(done))
                                     / static_cast<double>(total));
    if (pct != last_pct) {
        std::cerr << '\r' << "  progress: " << done << " / " << total
                  << " (" << pct << "%)" << std::flush;
        last_pct = pct;
    }
    if (done == total) std::cerr << '\n';
}

/// Slice the dataset to the first `max_days` snapshots. Used for
/// smoke runs that do not need the full archive loaded.
ore::marketdata::HistoricalDataset limit_days(
    const ore::marketdata::HistoricalDataset& full, std::size_t max_days)
{
    if (max_days == 0 || max_days >= full.size()) return full;
    std::vector<ore::marketdata::HistoricalSnapshot> trimmed;
    trimmed.reserve(max_days);
    for (std::size_t i = 0; i < max_days; ++i) {
        trimmed.push_back(full.snapshots()[i]);
    }
    return ore::marketdata::HistoricalDataset{
        std::string(full.ticker()), std::move(trimmed)};
}

} // namespace

int main(int argc, char** argv) try {
    using namespace ore::research;

    const std::filesystem::path root =
        (argc >= 2) ? argv[1] : "data/historical/spy";
    const std::string_view ticker =
        (argc >= 3) ? argv[2] : "SPY";
    const unsigned threads =
        (argc >= 4) ? static_cast<unsigned>(std::atoi(argv[3]))
                    : std::thread::hardware_concurrency();
    const std::size_t max_days =
        (argc >= 5) ? static_cast<std::size_t>(std::atoll(argv[4]))
                    : 0u;

    std::cout << "Loading historical archive from '"
              << root.string() << "' ...\n";
    const auto load_start = std::chrono::steady_clock::now();
    auto dataset = ore::marketdata::HistoricalLoader::load_from_eod(root, ticker);
    const auto load_end   = std::chrono::steady_clock::now();
    dataset = limit_days(dataset, max_days);

    const auto stats = dataset.stats();
    std::cout << "Loaded " << stats.trading_days << " trading days, "
              << stats.contracts_loaded << " contracts, in "
              << std::fixed << std::setprecision(2)
              << std::chrono::duration<double>(load_end - load_start).count()
              << "s.\n";
    std::cout << std::defaultfloat;

    if (dataset.empty()) {
        std::cerr << "No historical data found under '" << root.string()
                  << "'. Nothing to do.\n";
        return 0;
    }

    HistoricalResearchEngine::Config cfg{};
    cfg.threads    = threads == 0 ? 1u : threads;
    cfg.output_dir = "data/generated/research";
    cfg.progress   = &render_progress;

    HistoricalResearchEngine engine{dataset, cfg};

    std::cout << "\nRunning HistoricalCalibrationStudy\n";
    HistoricalCalibrationStudy study{};
    const auto report = engine.run(study);
    report.print(std::cout);

    const auto& s = study.stats();
    std::cout << "  Days processed         : " << s.days_processed         << '\n';
    std::cout << "  Contracts seen         : " << s.total_contracts        << '\n';
    std::cout << "  Contracts calibrated   : " << s.total_calibrated       << '\n';
    std::cout << "  Contracts skipped      : " << s.total_skipped          << '\n';
    std::cout << "  Solver failures        : " << s.total_failed_solves    << '\n';
    std::cout << "  Provider IV comparisons: " << s.total_iv_comparisons   << '\n';
    std::cout << "  Mean convergence rate  : " << s.mean_convergence_rate  << '\n';
    std::cout << "  Mean |IV error|        : " << s.mean_absolute_iv_error << '\n';

    std::cout << "\nDone. CSV outputs under '" << cfg.output_dir.string() << "/'.\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "historical_calibration: " << e.what() << "\n";
    return 1;
}
