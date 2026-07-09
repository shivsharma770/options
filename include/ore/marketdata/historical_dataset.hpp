/**
 * @file historical_dataset.hpp
 * @brief `HistoricalDataset` — an ordered collection of
 *        `HistoricalSnapshot`s for a single ticker, plus the
 *        `DatasetStatistics` summary type.
 *
 * A dataset is the primary object that empirical research code
 * operates on. Its ordering invariant (snapshots sorted by ascending
 * date) is enforced at construction — callers can rely on
 * `dataset[i].date() < dataset[i+1].date()` without re-sorting.
 *
 * The class is intentionally a simple container. It does not know how
 * data was produced (see `HistoricalLoader`) or how it will be
 * analysed (see `ore::analytics`). Keeping it dumb makes it usable
 * from all downstream milestones without dependency inversions.
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <ore/marketdata/historical_snapshot.hpp>

namespace ore::marketdata {

/**
 * @brief Summary statistics describing a loaded `HistoricalDataset`.
 *
 * Deliberately a plain-data struct with all fields public: consumers
 * (docs, CSV export, CI checks) routinely want to project it into
 * whatever output format they need.
 *
 * Fields marked "loader-only" are populated by
 * `HistoricalLoader::LoadResult` and threaded into the statistics by
 * whichever code owns the parse-failure list — see
 * `HistoricalDataset::stats()` for what the dataset can compute on its
 * own.
 */
struct DatasetStatistics {
    /** Ticker the dataset covers. */
    std::string ticker;
    /** Total number of snapshots loaded. */
    std::size_t trading_days{0};
    /** Aggregate number of contracts across all snapshots. */
    std::size_t contracts_loaded{0};
    /** Sum of contract counts divided by `trading_days`, or 0 if
     *  `trading_days == 0`. */
    double      average_contracts_per_day{0.0};
    /** Earliest snapshot date (nullopt if empty). */
    std::optional<std::chrono::year_month_day> first_date{};
    /** Latest snapshot date (nullopt if empty). */
    std::optional<std::chrono::year_month_day> last_date{};

    /** Loader-only: number of directories that were enumerated but
     *  failed to parse. Zero for datasets loaded strictly (where any
     *  failure would have thrown before the statistics were produced). */
    std::size_t parse_failures{0};

    /** Loader-only: business days in [first_date, last_date] that
     *  neither loaded nor failed — i.e. weekends and calendar gaps.
     *  Populated by `HistoricalLoader` because computing it requires
     *  the list of directories the loader saw, which the dataset
     *  itself does not retain. Left empty by `stats()`. */
    std::vector<std::chrono::year_month_day> missing_dates{};

    /** Human-readable multi-line dump — used by the example driver
     *  and by test failures where GoogleTest's default struct printer
     *  is not helpful. */
    void print(std::ostream& os) const;
};

/**
 * @class HistoricalDataset
 * @brief Ordered collection of `HistoricalSnapshot`s for one ticker.
 */
class HistoricalDataset {
public:
    /**
     * Construct from a ticker and a vector of snapshots.
     *
     * @param ticker    Ticker symbol (e.g. "SPY"). Not validated.
     * @param snapshots Snapshots to hold. **Sorted in-place** by
     *                  ascending date so `HistoricalDataset` can
     *                  guarantee monotone iteration. Duplicate dates
     *                  are permitted (the loader normally prevents
     *                  them, but two providers might disagree — the
     *                  container does not judge).
     */
    HistoricalDataset(std::string ticker,
                      std::vector<HistoricalSnapshot> snapshots);

    /** Ticker string this dataset was constructed with. */
    [[nodiscard]] const std::string& ticker() const noexcept { return ticker_; }

    /** All snapshots, in ascending date order. */
    [[nodiscard]] const std::vector<HistoricalSnapshot>& snapshots() const noexcept {
        return snapshots_;
    }

    /** Number of snapshots (== `trading_days` in the statistics). */
    [[nodiscard]] std::size_t size()  const noexcept { return snapshots_.size();  }
    /** True if there are no snapshots. */
    [[nodiscard]] bool        empty() const noexcept { return snapshots_.empty(); }

    /** Element access by index; UB out-of-range (mirror of
     *  `std::vector::operator[]`). Use `snapshots().at(i)` for
     *  bounds-checked access. */
    [[nodiscard]] const HistoricalSnapshot& operator[](std::size_t i) const noexcept {
        return snapshots_[i];
    }

    /** Bounds-checked access. Throws `std::out_of_range` if `i >= size()`. */
    [[nodiscard]] const HistoricalSnapshot& at(std::size_t i) const {
        return snapshots_.at(i);
    }

    /** Range-based-for over snapshots. */
    [[nodiscard]] auto begin() const noexcept { return snapshots_.begin(); }
    [[nodiscard]] auto end()   const noexcept { return snapshots_.end();   }

    /**
     * @brief First snapshot (undefined behaviour if `empty()` — mirrors
     *        `std::vector::front`).
     *
     * Prefer `first_date()` if the caller only needs the date; that
     * accessor is `nullopt`-safe. `front()` returns a full
     * `HistoricalSnapshot` reference for callers that want the whole
     * chain in one line.
     */
    [[nodiscard]] const HistoricalSnapshot& front() const noexcept {
        return snapshots_.front();
    }

    /** @brief Last snapshot (undefined behaviour if `empty()`). */
    [[nodiscard]] const HistoricalSnapshot& back()  const noexcept {
        return snapshots_.back();
    }

    /** First snapshot's date (nullopt if empty). */
    [[nodiscard]] std::optional<std::chrono::year_month_day> first_date() const noexcept;
    /** Last  snapshot's date (nullopt if empty). */
    [[nodiscard]] std::optional<std::chrono::year_month_day> last_date()  const noexcept;

    /**
     * @brief `true` iff a snapshot with the given date is present.
     *
     * O(log N) — binary search on the sorted snapshot vector. Callers
     * looking up many dates in the same dataset should prefer
     * `between(...)` if their targets are contiguous, which is a
     * single O(log N) plus a linear scan of the range.
     */
    [[nodiscard]] bool contains(std::chrono::year_month_day date) const noexcept;

    /**
     * @brief Fetch the snapshot for a specific date.
     *
     * @throws std::out_of_range if no snapshot for `date` exists.
     *         Callers who want a nullable result can use
     *         `find(date)`.
     */
    [[nodiscard]] const HistoricalSnapshot& at(std::chrono::year_month_day date) const;

    /**
     * @brief Fetch the snapshot for a date, or `nullptr` if absent.
     *
     * O(log N). Non-throwing counterpart to `at(date)`.
     */
    [[nodiscard]] const HistoricalSnapshot*
    find(std::chrono::year_month_day date) const noexcept;

    /**
     * @brief Return a zero-copy view over the snapshots whose dates
     *        fall in `[start, end]` (inclusive on both ends).
     *
     * Snapshots are already stored contiguously and in ascending date
     * order, so the returned `std::span` names a contiguous slice of
     * the underlying vector — no allocation, no copies. The span is
     * valid for as long as the dataset lives.
     *
     * If `start > end` or the range does not overlap the dataset, an
     * empty span is returned.
     */
    [[nodiscard]] std::span<const HistoricalSnapshot>
    between(std::chrono::year_month_day start,
            std::chrono::year_month_day end) const noexcept;

    /**
     * @brief Return a new `HistoricalDataset` containing only the
     *        snapshots for which `predicate` returns `true`.
     *
     * Uses copies — the returned dataset owns its snapshots. Prefer
     * `for_each` or `between` when the caller only needs to iterate
     * over the matches; both avoid allocating a new dataset.
     */
    [[nodiscard]] HistoricalDataset filter(
        const std::function<bool(const HistoricalSnapshot&)>& predicate) const;

    /**
     * @brief Apply `fn` to every snapshot in ascending-date order.
     *
     * Convenience wrapper around `std::for_each`. Provided for API
     * parity with the research spec — callers can equivalently write
     * `for (const auto& s : dataset)`.
     */
    template <typename F>
    void for_each(F&& fn) const {
        for (const auto& s : snapshots_) fn(s);
    }

    /**
     * Compute in-memory summary statistics. Fields that require
     * loader-side knowledge (parse failures, missing calendar dates)
     * are left at their defaults — the loader is responsible for
     * merging its own diagnostics on top.
     */
    [[nodiscard]] DatasetStatistics stats() const;

private:
    std::string                       ticker_;
    std::vector<HistoricalSnapshot>   snapshots_;
};

} // namespace ore::marketdata
