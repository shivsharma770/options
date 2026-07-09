/**
 * @file research_study.hpp
 * @brief Abstract base class for every study runnable by
 *        `HistoricalResearchEngine`.
 *
 * ### Contract
 *
 * A study implements four hooks:
 *
 *   - `begin(ctx)` — called once, before the first day. `ctx.day_index`
 *     is `0` and `ctx.snapshot()` refers to the first day (so studies
 *     that need the first day's spot for normalisation can grab it).
 *   - `process(ctx)` — called once per trading day, in ascending date
 *     order. The study inspects `ctx.chain()` (or `ctx.snapshot()`),
 *     runs whatever analytics it wants, and accumulates state.
 *   - `end(ctx, report)` — called once, after the last day. `ctx`
 *     refers to the *final* day; `report` is the (partially-filled)
 *     `ResearchReport` the engine will return. The study fills in
 *     the CSV paths it wrote, any warnings, and its own counters.
 *   - `name()` — a short stable identifier used in log lines, error
 *     messages, and default output filenames.
 *
 * ### Threading
 *
 * When `HistoricalResearchEngine::Config::threads > 1`, the engine
 * calls `clone()` once per worker thread. Each worker owns its own
 * clone and calls `begin`/`process`/`end` on it independently. When
 * every worker finishes, `merge(other)` is called on the primary
 * study for each worker's clone. Studies that support parallel
 * execution must implement `clone()` and `merge()` so:
 *
 *   - Two independent workers processing disjoint day-ranges, then
 *     merging back into the primary, must produce *bit-identical*
 *     results to a single-threaded run over the same days in the
 *     same order.
 *
 * A study that returns `nullptr` from `clone()` is treated as
 * non-parallelisable and the engine transparently falls back to
 * single-threaded execution — a safe default for prototypes and for
 * studies whose accumulator is order-dependent.
 *
 * ### Ownership
 *
 * The engine takes a `ResearchStudy&` — the study lives on the
 * caller's stack (or heap, whichever). This matches the milestone
 * spec's `engine.run(IVValidationStudy{})` idiom while still letting
 * long-lived studies persist across many runs.
 */
#pragma once

#include <memory>
#include <string_view>

#include <ore/research/research_context.hpp>
#include <ore/research/research_report.hpp>

namespace ore::research {

/**
 * @class ResearchStudy
 * @brief Abstract base of every runnable research task.
 */
class ResearchStudy {
public:
    virtual ~ResearchStudy() = default;

    /** Short human-readable identifier (e.g. `"IVValidationStudy"`). */
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    /**
     * @brief One-time pre-loop hook.
     *
     * @param ctx  Context referring to the *first* day the engine
     *             will dispatch, so studies that need to size
     *             accumulators to `ctx.total_days()` can do so here.
     */
    virtual void begin(const ResearchContext& ctx) { (void)ctx; }

    /**
     * @brief Per-day processing hook. Pure virtual — subclasses must
     *        override.
     *
     * @param ctx  Context for the day being processed. Guaranteed
     *             to reference a valid snapshot for the duration of
     *             the call and no longer.
     */
    virtual void process(const ResearchContext& ctx) = 0;

    /**
     * @brief One-time post-loop hook. The study writes CSVs, records
     *        any final warnings/counters, and populates the
     *        `report`.
     *
     * @param ctx     Context referring to the *last* day processed.
     * @param report  Partially-filled report the engine will return.
     *                Timing and progress counters are already set;
     *                the study appends `generated_files`, updates
     *                `processed_contracts` / `skipped_contracts`,
     *                and adds warnings.
     */
    virtual void end(const ResearchContext& ctx, ResearchReport& report) {
        (void)ctx;
        (void)report;
    }

    /**
     * @brief Produce a clone of this study for a parallel worker.
     *
     * Default returns `nullptr`, which disables parallel execution
     * (the engine falls back to running this exact instance
     * single-threaded). Studies that support parallelism must
     * override so:
     *
     *   - The clone is a fresh instance with an empty accumulator.
     *   - Its configuration matches `*this` exactly.
     *   - Its state can later be merged back via `merge(clone)`.
     */
    [[nodiscard]] virtual std::unique_ptr<ResearchStudy> clone() const {
        return nullptr;
    }

    /**
     * @brief Merge the state of a worker clone back into this study.
     *
     * Only called on the primary study, and only for clones the
     * engine received from this study's `clone()`. The default is a
     * no-op, which is fine for studies that never opt into parallel
     * execution (their `clone()` returns `nullptr`).
     */
    virtual void merge(const ResearchStudy& other) { (void)other; }

protected:
    ResearchStudy() = default;
    ResearchStudy(const ResearchStudy&) = default;
    ResearchStudy(ResearchStudy&&) noexcept = default;
    ResearchStudy& operator=(const ResearchStudy&) = default;
    ResearchStudy& operator=(ResearchStudy&&) noexcept = default;
};

} // namespace ore::research
