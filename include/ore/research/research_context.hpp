/**
 * @file research_context.hpp
 * @brief Read-only handle passed to a `ResearchStudy` for each trading day.
 *
 * `ResearchContext` is deliberately a *view*: every member is a
 * reference or a small POD, no ownership, no allocation. The engine
 * constructs one on the stack per `process()` invocation and hands it
 * to the study. That keeps the hot loop allocation-free even when
 * running a study over years of daily data.
 *
 * ### What the context exposes
 *
 * The context wraps the "here is today's data" surface a study needs:
 *
 *  - The `HistoricalSnapshot` and its underlying `OptionChain`
 *    (functionally equivalent, but exposing both lets studies pick
 *    whichever type feels more natural — they iterate over identical
 *    contents).
 *  - `MarketSnapshot` and `Underlying` shortcut accessors for the
 *    common cases of "give me spot" and "give me the ticker" without
 *    reaching through `snapshot().chain().market()`.
 *  - A `date()` shortcut.
 *  - Progress metadata (`day_index`, `total_days`) — occasionally
 *    useful in `end()` reports.
 *  - `output_dir` for CSV exports (so studies never hardcode a path).
 *
 * ### Thread-safety
 *
 * `ResearchContext` carries only references to data that the engine
 * guarantees to outlive the context, so multiple threads may read the
 * same context concurrently. When the engine runs in parallel mode
 * each worker thread receives its own context; nothing is shared
 * across worker boundaries.
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>

#include <ore/core/market_snapshot.hpp>
#include <ore/core/option_chain.hpp>
#include <ore/core/underlying.hpp>
#include <ore/marketdata/historical_snapshot.hpp>

namespace ore::research {

/**
 * @class ResearchContext
 * @brief View of one trading day's data plus per-run metadata.
 */
class ResearchContext {
public:
    /**
     * @brief Construct from a snapshot, day index, total count, and
     *        output directory. All parameters are held by reference
     *        or by value — the caller keeps ownership of the
     *        snapshot.
     */
    ResearchContext(const ore::marketdata::HistoricalSnapshot& snapshot,
                    std::size_t day_index,
                    std::size_t total_days,
                    const std::filesystem::path& output_dir) noexcept
        : snapshot_(snapshot),
          day_index_(day_index),
          total_days_(total_days),
          output_dir_(output_dir) {}

    /** @brief Today's historical snapshot (date + chain). */
    [[nodiscard]] const ore::marketdata::HistoricalSnapshot& snapshot() const noexcept {
        return snapshot_;
    }

    /** @brief Today's option chain — a shortcut for `snapshot().chain()`. */
    [[nodiscard]] const ore::core::OptionChain& chain() const noexcept {
        return snapshot_.chain();
    }

    /** @brief Today's market state (spot, rate, dividend yield, ...). */
    [[nodiscard]] const ore::core::MarketSnapshot& market() const noexcept {
        return snapshot_.market();
    }

    /** @brief Underlying identity (ticker, etc.). */
    [[nodiscard]] const ore::core::Underlying& underlying() const noexcept {
        return snapshot_.underlying();
    }

    /** @brief Calendar date of today's snapshot. */
    [[nodiscard]] std::chrono::year_month_day date() const noexcept {
        return snapshot_.date();
    }

    /** @brief Spot price observed today. */
    [[nodiscard]] double spot() const noexcept { return market().spot; }

    /** @brief Zero-based index of `snapshot` in the engine's dataset. */
    [[nodiscard]] std::size_t day_index()  const noexcept { return day_index_; }

    /** @brief Total number of trading days the engine is iterating over. */
    [[nodiscard]] std::size_t total_days() const noexcept { return total_days_; }

    /**
     * @brief Directory the study should write generated CSVs into.
     *
     * Never hardcoded in any study — flows down from
     * `HistoricalResearchEngine::Config::output_dir` (which defaults
     * to `data/generated/research/`).
     */
    [[nodiscard]] const std::filesystem::path& output_dir() const noexcept {
        return output_dir_;
    }

private:
    const ore::marketdata::HistoricalSnapshot& snapshot_;
    std::size_t                                day_index_;
    std::size_t                                total_days_;
    const std::filesystem::path&               output_dir_;
};

} // namespace ore::research
