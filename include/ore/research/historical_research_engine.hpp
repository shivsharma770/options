/**
 * @file historical_research_engine.hpp
 * @brief Orchestrator that runs a `ResearchStudy` over a
 *        `HistoricalDataset`.
 *
 * ### Responsibilities
 *
 *   - Iterate the dataset in ascending-date order.
 *   - For each snapshot: build a `ResearchContext` and call
 *     `study.process(ctx)`.
 *   - Manage bookkeeping: timing, error trapping, progress callback,
 *     final report assembly.
 *   - Support optional parallel execution via `clone()` + `merge()`
 *     on the study.
 *
 * ### Determinism
 *
 * `HistoricalResearchEngine` is designed so that parallel execution
 * produces **numerically identical** results to serial execution,
 * provided the study implements `clone()` and `merge()` correctly.
 * The engine partitions the dataset into contiguous date ranges and
 * assigns one range per worker; within a range each worker sees the
 * days in ascending order. The primary study accumulates the
 * workers' state in worker index order (thread `0`, then `1`, ...),
 * which for the built-in studies is enough because their
 * accumulators are order-independent (vectors of rows, later
 * sorted; sum-and-count aggregates; ...).
 *
 * ### Not the engine's job
 *
 *   - Deciding what "processed" means — the study defines that.
 *   - Filtering the dataset — do that in `HistoricalDataset::filter`.
 *   - Writing anything to disk — the study writes its CSVs in
 *     `end()`, choosing filenames based on `ResearchContext::output_dir`.
 */
#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <type_traits>
#include <utility>

#include <ore/marketdata/historical_dataset.hpp>
#include <ore/research/research_context.hpp>
#include <ore/research/research_report.hpp>
#include <ore/research/research_study.hpp>

namespace ore::research {

/**
 * @class HistoricalResearchEngine
 * @brief Runs `ResearchStudy`s over a `HistoricalDataset`.
 */
class HistoricalResearchEngine {
public:
    /**
     * @brief Configuration for a run.
     *
     * Default-constructed `Config{}` runs single-threaded and writes
     * outputs to `data/generated/research/`. No study or dataset
     * knows about this default explicitly — everything flows through
     * this struct.
     */
    struct Config {
        /**
         * Worker threads to use. `1` is single-threaded and always
         * safe. Values greater than 1 require the study to opt in
         * via a working `clone()` / `merge()` pair; if `clone()`
         * returns `nullptr`, the engine falls back to
         * single-threaded execution silently. `0` is treated as `1`.
         *
         * `std::thread::hardware_concurrency()` is the typical
         * "use everything" choice.
         */
        unsigned threads{1};

        /**
         * Directory under which the study writes CSV outputs. Every
         * study defaults its own filenames relative to this path;
         * pass a different value here to redirect outputs (e.g. a
         * temp directory for unit tests).
         */
        std::filesystem::path output_dir{"data/generated/research"};

        /**
         * Optional progress callback. Called after each processed
         * day with `(processed_days, total_days)`. In parallel
         * runs the callback is serialised through an internal
         * mutex, so it may be called from any worker thread but
         * never concurrently. Pass an empty `std::function` (the
         * default) to disable.
         */
        std::function<void(std::size_t /*processed*/,
                           std::size_t /*total*/)> progress{};

        /**
         * If `true`, the engine trap and swallow exceptions thrown
         * from `study.process()` into `ResearchReport::errors` and
         * continues with the next day. If `false`, the exception is
         * propagated out of `run()` and the run is aborted. The
         * default is `true` — long-running research jobs generally
         * prefer a partial result to a total failure.
         */
        bool continue_on_error{true};

        /**
         * If `true`, the engine creates `output_dir` (recursively)
         * before dispatching. Studies still take care of their own
         * subdirectories if they need any. Turn off if the caller
         * wants to pre-create with different permissions.
         */
        bool create_output_dir{true};
    };

    /**
     * @brief The default `Config` used when a caller omits it.
     *
     * Defined out-of-line so the constructor's default argument can be
     * a plain function call (`= default_config()`) rather than an
     * in-class `Config{}` braced-init. GCC rejects the braced form here
     * because it would evaluate `Config`'s default member initializers
     * before this enclosing class is complete; the call defers that to
     * the (already-complete) point of definition.
     */
    [[nodiscard]] static Config default_config();

    /**
     * @brief Construct an engine over an existing dataset.
     *
     * @param dataset  Dataset to iterate. Held by const reference —
     *                 the caller keeps ownership and must outlive
     *                 the engine.
     * @param config   Run configuration. Copied.
     */
    HistoricalResearchEngine(const ore::marketdata::HistoricalDataset& dataset,
                             Config config = default_config()) noexcept
        : dataset_(dataset), config_(std::move(config)) {}

    /**
     * @brief Run a study over every day in the dataset.
     *
     * Accepts both lvalues and rvalues (including the milestone
     * spec's `engine.run(IVValidationStudy{})` idiom) via a
     * forwarding reference. Whatever the caller passes, the engine
     * treats as a base `ResearchStudy&` internally.
     *
     * @tparam StudyT  Any type derived (publicly) from
     *                 `ResearchStudy`.
     * @param study    The study to execute. Its `begin` /
     *                 `process` / `end` hooks are invoked as
     *                 described in `ResearchStudy`.
     *
     * @return `ResearchReport` summarising the run.
     */
    template <typename StudyT>
        requires std::is_base_of_v<ResearchStudy, std::remove_cvref_t<StudyT>>
    ResearchReport run(StudyT&& study) {
        // A forwarding-ref parameter is always an lvalue inside the
        // function, so binding to `ResearchStudy&` here is safe for
        // both lvalue and rvalue callers.
        ResearchStudy& ref = study;
        return run_dispatch(ref);
    }

    /** @brief Immutable access to the dataset the engine wraps. */
    [[nodiscard]] const ore::marketdata::HistoricalDataset& dataset() const noexcept {
        return dataset_;
    }

    /** @brief Immutable access to the run configuration. */
    [[nodiscard]] const Config& config() const noexcept { return config_; }

private:
    const ore::marketdata::HistoricalDataset& dataset_;
    Config                                    config_;

    /** Dispatch on `config_.threads` and the study's `clone()`
     *  capability. Kept out-of-line so the header does not include
     *  `<thread>`, `<mutex>`, etc. */
    ResearchReport run_dispatch(ResearchStudy& study);

    /** Serial fast path used when `threads <= 1` or the study is not
     *  parallelisable. */
    ResearchReport run_serial(ResearchStudy& study);

    /** Parallel fast path used when `threads > 1` **and** the study
     *  returns a non-null `clone()`. */
    ResearchReport run_parallel(ResearchStudy& study, unsigned threads);
};

} // namespace ore::research
