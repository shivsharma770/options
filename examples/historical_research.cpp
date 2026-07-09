/**
 * @file historical_research.cpp
 * @brief End-to-end demonstration of `HistoricalResearchEngine`.
 *
 * Loads the SPY historical archive from `data/historical/spy/`,
 * runs the three built-in studies (`IVValidationStudy`,
 * `GreeksValidationStudy`, `PricingValidationStudy`) against it, and
 * prints per-study reports.  Every CSV output lands beneath
 * `data/generated/research/`.
 *
 * Usage
 * -----
 *
 *     ./example_historical_research [root] [ticker] [threads] [max_days]
 *
 * Defaults
 *   root     = data/historical/spy
 *   ticker   = SPY
 *   threads  = std::thread::hardware_concurrency()
 *   max_days = 0 (all days on disk)
 *
 * `max_days > 0` slices the dataset to its first `max_days` snapshots
 * — handy for a smoke run without loading the full 2010-2021
 * archive.
 */

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <ore/marketdata/historical_dataset.hpp>
#include <ore/marketdata/historical_loader.hpp>
#include <ore/marketdata/historical_snapshot.hpp>
#include <ore/research/greeks_validation_study.hpp>
#include <ore/research/historical_research_engine.hpp>
#include <ore/research/iv_validation_study.hpp>
#include <ore/research/pricing_validation_study.hpp>
#include <ore/research/research_report.hpp>

namespace {

/// Simple stderr progress bar. Renders "processed / total" every
/// time the callback fires; on a modern terminal this stays on the
/// same line thanks to `\r`.
void render_progress(std::size_t done, std::size_t total) {
    static int last_pct = -1;
    const int pct = static_cast<int>((100.0 * static_cast<double>(done))
                                     / static_cast<double>(total));
    if (pct != last_pct) {
        std::cerr << '\r' << "  progress: " << done << " / " << total
                  << " (" << pct << "%)" << std::flush;
        last_pct = pct;
    }
    if (done == total) std::cerr << '\n';
}

/// Trim the dataset to the first N snapshots. Used only when the
/// caller asked for a smoke run.
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

    // Study 1: IV validation.
    std::cout << "\n[1/3] IVValidationStudy\n";
    IVValidationStudy iv{};
    auto iv_report = engine.run(iv);
    iv_report.print(std::cout);
    std::cout << "  Mean |IV error|    : " << iv.stats().mean_absolute_error << '\n';
    std::cout << "  Median |IV error|  : " << iv.stats().median_absolute_error << '\n';
    std::cout << "  IV RMSE            : " << iv.stats().rmse << '\n';
    std::cout << "  Worst |IV error|   : " << iv.stats().worst_error << '\n';
    std::cout << "  Convergence rate   : " << iv.stats().convergence_rate << '\n';

    // Study 2: Greeks validation.
    std::cout << "\n[2/3] GreeksValidationStudy\n";
    GreeksValidationStudy greeks{};
    auto greeks_report = engine.run(greeks);
    greeks_report.print(std::cout);
    std::cout << "  Delta MAE / RMSE   : "
              << greeks.stats().delta.mae << " / " << greeks.stats().delta.rmse << '\n';
    std::cout << "  Gamma MAE / RMSE   : "
              << greeks.stats().gamma.mae << " / " << greeks.stats().gamma.rmse << '\n';
    std::cout << "  Theta MAE / RMSE   : "
              << greeks.stats().theta.mae << " / " << greeks.stats().theta.rmse << '\n';
    std::cout << "  Vega  MAE / RMSE   : "
              << greeks.stats().vega.mae  << " / " << greeks.stats().vega.rmse  << '\n';
    std::cout << "  Rho   MAE / RMSE   : "
              << greeks.stats().rho.mae   << " / " << greeks.stats().rho.rmse   << '\n';

    // Study 3: Pricing round-trip.
    std::cout << "\n[3/3] PricingValidationStudy\n";
    PricingValidationStudy pv{};
    auto pv_report = engine.run(pv);
    pv_report.print(std::cout);
    std::cout << "  Mean |residual|    : " << pv.stats().mean_abs_residual  << '\n';
    std::cout << "  RMSE residual      : " << pv.stats().rmse_residual      << '\n';
    std::cout << "  Worst |residual|   : " << pv.stats().worst_abs_residual << '\n';
    std::cout << "  Fraction < tol     : " << pv.stats().fraction_within_tolerance << '\n';

    std::cout << "\nDone. CSV outputs under '" << cfg.output_dir.string() << "/'.\n";
    return 0;

} catch (const std::exception& e) {
    std::cerr << "historical_research: " << e.what() << "\n";
    return 1;
}
