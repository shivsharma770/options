#include <ore/research/historical_research_engine.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <ore/marketdata/historical_dataset.hpp>
#include <ore/marketdata/historical_snapshot.hpp>
#include <ore/research/research_context.hpp>
#include <ore/research/research_report.hpp>
#include <ore/research/research_study.hpp>

namespace ore::research {

namespace {

namespace fs = std::filesystem;

/// Ensure `dir` exists on disk (creating parents as needed). Any
/// filesystem error becomes an entry in `report.errors` rather than
/// an exception — a research run should not fail because of a
/// permissions glitch on the output directory.
void ensure_output_dir(const fs::path& dir, ResearchReport& report) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        std::ostringstream oss;
        oss << "cannot create output directory '" << dir.string()
            << "': " << ec.message();
        report.errors.push_back(oss.str());
    }
}

/// Format a study exception into `report.errors`. Extracted so the
/// serial and parallel paths report failures identically.
void record_exception(std::size_t day_index,
                      const std::exception& e,
                      ResearchReport& report,
                      std::mutex* mutex)
{
    std::ostringstream oss;
    oss << "day[" << day_index << "]: " << e.what();
    if (mutex) {
        std::lock_guard<std::mutex> lock(*mutex);
        report.errors.push_back(oss.str());
    } else {
        report.errors.push_back(oss.str());
    }
}

/// Same as `record_exception` for unknown-type throws. The two
/// helpers exist so the `catch(...)` branch does not fabricate a
/// misleading `.what()` value.
void record_unknown(std::size_t day_index,
                    ResearchReport& report,
                    std::mutex* mutex)
{
    std::ostringstream oss;
    oss << "day[" << day_index << "]: unknown exception";
    if (mutex) {
        std::lock_guard<std::mutex> lock(*mutex);
        report.errors.push_back(oss.str());
    } else {
        report.errors.push_back(oss.str());
    }
}

/// Process `snapshot` under `study`, catching exceptions per the
/// engine's `continue_on_error` policy.
///
/// Returns `true` if execution should continue to the next day.
[[nodiscard]] bool
process_one(ResearchStudy& study,
            const ore::marketdata::HistoricalSnapshot& snapshot,
            std::size_t day_index,
            std::size_t total_days,
            const fs::path& output_dir,
            bool continue_on_error,
            ResearchReport& report,
            std::mutex* error_mutex)
{
    const ResearchContext ctx{snapshot, day_index, total_days, output_dir};
    try {
        study.process(ctx);
    } catch (const std::exception& e) {
        record_exception(day_index, e, report, error_mutex);
        if (!continue_on_error) return false;
    } catch (...) {
        record_unknown(day_index, report, error_mutex);
        if (!continue_on_error) return false;
    }
    return true;
}

} // namespace

ResearchReport HistoricalResearchEngine::run_dispatch(ResearchStudy& study) {
    unsigned threads = config_.threads == 0 ? 1u : config_.threads;

    // Parallel path is only used if there's real work to split
    // *and* the study opted in via a non-null clone(). We probe
    // clone() up front so the fallback is transparent.
    if (threads > 1u && dataset_.size() >= threads) {
        auto probe = study.clone();
        if (probe) {
            // Discard the probe — we'll create fresh clones inside
            // run_parallel. The probe is only used to check whether
            // the study supports the parallel path at all.
            return run_parallel(study, threads);
        }
    }
    return run_serial(study);
}

ResearchReport HistoricalResearchEngine::run_serial(ResearchStudy& study) {
    ResearchReport report{};
    report.study_name = std::string(study.name());

    if (config_.create_output_dir) {
        ensure_output_dir(config_.output_dir, report);
    }

    const auto& snapshots = dataset_.snapshots();
    const std::size_t total = snapshots.size();
    const auto start_wall = std::chrono::steady_clock::now();

    if (total > 0) {
        // begin() sees the first day.
        const ResearchContext begin_ctx{
            snapshots.front(), 0, total, config_.output_dir};
        try {
            study.begin(begin_ctx);
        } catch (const std::exception& e) {
            record_exception(0, e, report, nullptr);
            if (!config_.continue_on_error) {
                report.runtime_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start_wall).count();
                return report;
            }
        }
    }

    for (std::size_t i = 0; i < total; ++i) {
        if (!process_one(study, snapshots[i], i, total,
                         config_.output_dir,
                         config_.continue_on_error,
                         report, nullptr))
        {
            break;
        }
        ++report.processed_days;
        if (config_.progress) config_.progress(report.processed_days, total);
    }

    if (total > 0) {
        // end() sees the last day.
        const ResearchContext end_ctx{
            snapshots.back(), total - 1, total, config_.output_dir};
        try {
            study.end(end_ctx, report);
        } catch (const std::exception& e) {
            record_exception(total - 1, e, report, nullptr);
        }
    }

    report.runtime_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_wall).count();
    return report;
}

ResearchReport HistoricalResearchEngine::run_parallel(ResearchStudy& study,
                                                     unsigned threads)
{
    ResearchReport report{};
    report.study_name = std::string(study.name());

    if (config_.create_output_dir) {
        ensure_output_dir(config_.output_dir, report);
    }

    const auto& snapshots = dataset_.snapshots();
    const std::size_t total = snapshots.size();
    const auto start_wall = std::chrono::steady_clock::now();

    if (total == 0) {
        report.runtime_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_wall).count();
        return report;
    }

    // begin() runs once on the primary study — same as the serial
    // path. Cloning happens *after* begin() so the child accumulators
    // start empty regardless of what begin() did on the parent.
    {
        const ResearchContext begin_ctx{
            snapshots.front(), 0, total, config_.output_dir};
        try {
            study.begin(begin_ctx);
        } catch (const std::exception& e) {
            record_exception(0, e, report, nullptr);
        }
    }

    // Partition the dataset into `threads` contiguous ranges. Using
    // contiguous ranges (rather than round-robin dispatch) gives each
    // thread cache-friendly access to `snapshots_`. It also keeps
    // per-thread progress monotone, which matters for progress
    // callbacks that display "day N of M".
    struct Chunk {
        std::size_t begin;
        std::size_t end;
    };
    std::vector<Chunk> chunks;
    chunks.reserve(threads);
    {
        const std::size_t base = total / threads;
        const std::size_t rem  = total % threads;
        std::size_t cursor = 0;
        for (unsigned t = 0; t < threads; ++t) {
            const std::size_t size = base + (t < rem ? 1u : 0u);
            chunks.push_back({cursor, cursor + size});
            cursor += size;
        }
    }

    // Per-worker cloned studies. Filled by the workers; merged into
    // the primary study once every worker has joined.
    std::vector<std::unique_ptr<ResearchStudy>> clones(threads);

    std::mutex error_mutex;
    std::mutex progress_mutex;
    std::atomic<std::size_t> completed_days{0};

    auto worker = [&](unsigned tid) {
        auto clone = study.clone();
        if (!clone) return; // defensive — dispatch checked this already
        const Chunk c = chunks[tid];

        for (std::size_t i = c.begin; i < c.end; ++i) {
            if (!process_one(*clone, snapshots[i], i, total,
                             config_.output_dir,
                             config_.continue_on_error,
                             report, &error_mutex))
            {
                break;
            }
            const auto now_done = completed_days.fetch_add(1) + 1;
            if (config_.progress) {
                std::lock_guard<std::mutex> lock(progress_mutex);
                config_.progress(now_done, total);
            }
        }
        clones[tid] = std::move(clone);
    };

    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back(worker, t);
    }
    for (auto& th : pool) th.join();

    // Merge clones back into the primary study in *deterministic*
    // worker order (thread 0, 1, 2, ...). This matches what the
    // serial run would have done for the built-in studies because
    // their accumulators are append-only vectors (order-preserving)
    // or additive counters (associative & commutative).
    for (auto& clone : clones) {
        if (clone) study.merge(*clone);
    }

    report.processed_days = completed_days.load();

    {
        const ResearchContext end_ctx{
            snapshots.back(), total - 1, total, config_.output_dir};
        try {
            study.end(end_ctx, report);
        } catch (const std::exception& e) {
            record_exception(total - 1, e, report, nullptr);
        }
    }

    report.runtime_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_wall).count();
    return report;
}

} // namespace ore::research
