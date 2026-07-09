#include <ore/marketdata/historical_dataset.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iterator>
#include <optional>
#include <ostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ore::marketdata {

namespace {

/**
 * @brief Render `ymd` as `YYYY-MM-DD` for the `print()` diagnostic
 *        dump. Kept local to this TU because the same rendering shows
 *        up several times below.
 */
std::string format_date(std::chrono::year_month_day ymd) {
    std::ostringstream oss;
    oss << static_cast<int>(ymd.year()) << '-'
        << std::setw(2) << std::setfill('0')
        << static_cast<unsigned>(ymd.month()) << '-'
        << std::setw(2) << std::setfill('0')
        << static_cast<unsigned>(ymd.day());
    return oss.str();
}

}  // namespace

HistoricalDataset::HistoricalDataset(
    std::string ticker,
    std::vector<HistoricalSnapshot> snapshots)
    : ticker_(std::move(ticker)),
      snapshots_(std::move(snapshots))
{
    // Ordering invariant enforced at construction so consumers can
    // rely on monotone dates without having to re-sort. Stable sort
    // preserves relative order between hypothetical duplicate-date
    // snapshots (see the header for why we don't reject those).
    std::stable_sort(
        snapshots_.begin(), snapshots_.end(),
        [](const HistoricalSnapshot& a, const HistoricalSnapshot& b) {
            return std::chrono::sys_days{a.date()} < std::chrono::sys_days{b.date()};
        });
}

std::optional<std::chrono::year_month_day>
HistoricalDataset::first_date() const noexcept {
    if (snapshots_.empty()) return std::nullopt;
    return snapshots_.front().date();
}

std::optional<std::chrono::year_month_day>
HistoricalDataset::last_date() const noexcept {
    if (snapshots_.empty()) return std::nullopt;
    return snapshots_.back().date();
}

namespace {

/// Strict-weak-ordering on snapshots by date. Reused by the binary-
/// search accessors below.
struct SnapshotDateLess {
    bool operator()(const HistoricalSnapshot& s,
                    std::chrono::year_month_day d) const noexcept {
        return std::chrono::sys_days{s.date()} < std::chrono::sys_days{d};
    }
    bool operator()(std::chrono::year_month_day d,
                    const HistoricalSnapshot& s) const noexcept {
        return std::chrono::sys_days{d} < std::chrono::sys_days{s.date()};
    }
};

} // namespace

bool HistoricalDataset::contains(std::chrono::year_month_day date) const noexcept {
    return std::binary_search(snapshots_.begin(), snapshots_.end(),
                              date, SnapshotDateLess{});
}

const HistoricalSnapshot*
HistoricalDataset::find(std::chrono::year_month_day date) const noexcept {
    auto it = std::lower_bound(snapshots_.begin(), snapshots_.end(),
                               date, SnapshotDateLess{});
    if (it == snapshots_.end()) return nullptr;
    if (it->date() != date) return nullptr;
    return &*it;
}

const HistoricalSnapshot&
HistoricalDataset::at(std::chrono::year_month_day date) const {
    if (const auto* p = find(date)) return *p;
    std::ostringstream oss;
    oss << "HistoricalDataset::at(" << static_cast<int>(date.year()) << '-'
        << static_cast<unsigned>(date.month()) << '-'
        << static_cast<unsigned>(date.day())
        << "): no snapshot with that date";
    throw std::out_of_range(oss.str());
}

std::span<const HistoricalSnapshot>
HistoricalDataset::between(std::chrono::year_month_day start,
                           std::chrono::year_month_day end) const noexcept
{
    if (snapshots_.empty()) return {};
    if (std::chrono::sys_days{end} < std::chrono::sys_days{start}) return {};

    // lower_bound gives the first snapshot with date >= start.
    // upper_bound gives the first with date > end. The half-open
    // range in between is exactly what the span should cover.
    auto first = std::lower_bound(snapshots_.begin(), snapshots_.end(),
                                  start, SnapshotDateLess{});
    auto last  = std::upper_bound(snapshots_.begin(), snapshots_.end(),
                                  end, SnapshotDateLess{});
    if (first >= last) return {};

    const auto begin_idx = static_cast<std::size_t>(std::distance(snapshots_.begin(), first));
    const auto count     = static_cast<std::size_t>(std::distance(first, last));
    return std::span<const HistoricalSnapshot>(snapshots_.data() + begin_idx, count);
}

HistoricalDataset HistoricalDataset::filter(
    const std::function<bool(const HistoricalSnapshot&)>& predicate) const
{
    std::vector<HistoricalSnapshot> keep;
    keep.reserve(snapshots_.size());
    for (const auto& s : snapshots_) {
        if (predicate(s)) keep.push_back(s);
    }
    return HistoricalDataset{ticker_, std::move(keep)};
}

DatasetStatistics HistoricalDataset::stats() const {
    DatasetStatistics s{};
    s.ticker        = ticker_;
    s.trading_days  = snapshots_.size();
    s.first_date    = first_date();
    s.last_date     = last_date();

    std::size_t total_contracts = 0;
    for (const auto& snap : snapshots_) {
        total_contracts += snap.size();
    }
    s.contracts_loaded = total_contracts;

    // Guard against divide-by-zero on empty datasets. Reporting 0.0
    // is slightly ambiguous ("no data" vs. "data with zero contracts")
    // but tests always check `trading_days` alongside so the ambiguity
    // never bites in practice.
    if (s.trading_days > 0) {
        s.average_contracts_per_day =
            static_cast<double>(total_contracts) /
            static_cast<double>(s.trading_days);
    }

    // parse_failures and missing_dates are loader-side concepts; leave
    // them at their defaults. `HistoricalLoader::LoadResult` merges
    // them in before handing statistics to the user.
    return s;
}

void DatasetStatistics::print(std::ostream& os) const {
    os << "Ticker           : " << ticker << '\n';
    os << "Trading days     : " << trading_days << '\n';
    os << "Contracts loaded : " << contracts_loaded << '\n';
    os << "Avg contracts/day: " << average_contracts_per_day << '\n';
    if (first_date.has_value() && last_date.has_value()) {
        os << "Date range       : "
           << format_date(*first_date) << " ... "
           << format_date(*last_date)  << '\n';
    } else {
        os << "Date range       : (empty)\n";
    }
    os << "Parse failures   : " << parse_failures << '\n';
    os << "Missing dates    : " << missing_dates.size();
    // Cap the printed list — a several-year dataset can have hundreds
    // of weekend gaps and we don't want to drown the log.
    constexpr std::size_t kMaxPrinted = 5;
    if (!missing_dates.empty()) {
        os << " (first "
           << std::min(missing_dates.size(), kMaxPrinted) << ": ";
        for (std::size_t i = 0; i < std::min(missing_dates.size(), kMaxPrinted); ++i) {
            if (i > 0) os << ", ";
            os << format_date(missing_dates[i]);
        }
        os << ")";
    }
    os << '\n';
}

} // namespace ore::marketdata
