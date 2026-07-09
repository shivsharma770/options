/**
 * @file research_csv.hpp
 * @brief Tiny CSV-writing helpers shared by every research study.
 *
 * A single place to enforce the project-wide formatting conventions:
 *
 *   - ISO-8601 dates (`YYYY-MM-DD`).
 *   - `%.17g` floating-point precision (round-trip exact for
 *     IEEE-754 doubles).
 *   - Empty fields for missing optionals (unlike the "0" a naïve
 *     `<<` would print).
 *
 * The class is deliberately small — enough to write the built-in
 * studies' CSVs, no more. Consumers that want a fuller CSV writer
 * can build one on top of `<fstream>`.
 */
#pragma once

#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace ore::research {

/**
 * @class ResearchCsvWriter
 * @brief Line-buffered CSV writer with the project's formatting
 *        conventions baked in.
 *
 * ### Usage
 *
 * @code
 * ResearchCsvWriter w(path, {"date", "strike", "iv"});
 * w.write_row(
 *     w.date(row.date),
 *     w.number(row.strike),
 *     w.optional_number(row.iv));
 * @endcode
 *
 * `write_row` accepts any number of already-formatted strings and
 * joins them with a single comma. No embedded commas or newlines
 * are permitted in a field — asserting that upfront keeps the writer
 * simple; every built-in study only writes numerics + ISO dates so
 * this is a non-issue.
 */
class ResearchCsvWriter {
public:
    /**
     * @brief Open `path` for writing (overwriting) and emit the
     *        header row.
     *
     * @throws std::runtime_error if the file cannot be opened.
     */
    ResearchCsvWriter(const std::filesystem::path& path,
                      std::initializer_list<std::string_view> columns);

    /** @brief Close the stream. Flushes are automatic. */
    ~ResearchCsvWriter() = default;

    // Not copyable — an open ofstream has no sensible copy semantics.
    ResearchCsvWriter(const ResearchCsvWriter&) = delete;
    ResearchCsvWriter& operator=(const ResearchCsvWriter&) = delete;
    ResearchCsvWriter(ResearchCsvWriter&&) = default;
    ResearchCsvWriter& operator=(ResearchCsvWriter&&) = default;

    /**
     * @brief Format a `double` at `%.17g` precision — the minimum
     *        precision guaranteed to round-trip an IEEE-754 double
     *        exactly (see IEEE-754-2008 §5.12.2). Callers should
     *        prefer this over `std::to_string` (fixed 6 digits) and
     *        over `<<` (defaults to 6 or 15 depending on stream
     *        settings) for anything they intend to compare against
     *        the C++ layer later.
     */
    [[nodiscard]] static std::string number(double x);

    /**
     * @brief Format a `long long` — plain decimal, no thousands
     *        separators. Kept for parity with `number()` so studies
     *        never have to reach for `std::to_string` themselves.
     */
    [[nodiscard]] static std::string integer(long long x);
    [[nodiscard]] static std::string integer(std::size_t x);

    /**
     * @brief Format a `year_month_day` as ISO-8601 `YYYY-MM-DD`.
     */
    [[nodiscard]] static std::string date(std::chrono::year_month_day ymd);

    /**
     * @brief Format an optional double: empty string if `nullopt`,
     *        otherwise `%.17g`. Empty fields survive the CSV → pandas
     *        round-trip as `NaN` which is exactly the semantics
     *        every downstream analysis wants.
     */
    [[nodiscard]] static std::string optional_number(std::optional<double> v);

    /**
     * @brief Append a row of already-formatted fields. Fields are
     *        emitted verbatim — the caller is responsible for
     *        having called the formatting helpers above.
     */
    void write_row(std::initializer_list<std::string> fields);

    /** @brief Path the writer is writing to. */
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
    std::ofstream         out_;
};

} // namespace ore::research
