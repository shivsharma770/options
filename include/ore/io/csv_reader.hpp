/**
 * @file csv_reader.hpp
 * @brief Generic CSV parser. Knows nothing about finance.
 *
 * `ore::io::CsvTable` reads a CSV file (or in-memory string) into memory and
 * exposes rows as `CsvRow` handles that look up fields by column name.
 *
 * Scope
 * -----
 * - Header row is required by default (see `CsvOptions::has_header`).
 * - Fields separated by a configurable single-character delimiter.
 * - Fields may be enclosed in double quotes; an embedded double quote is
 *   escaped by doubling (`""`).
 * - Line terminators: `\n` or `\r\n`.
 * - **Not supported:** embedded newlines within quoted fields. yfinance /
 *   pandas output does not produce these and full RFC 4180 support is more
 *   parser than this milestone needs.
 * - Every failure (I/O, malformed row, wrong field count, non-numeric where
 *   a numeric was requested) throws `ore::io::CsvParseError` with a
 *   message naming the file, the line number, and the offending column
 *   where relevant.
 */
#pragma once

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ore::io {

/**
 * Options controlling CSV parsing behaviour.
 */
struct CsvOptions {
    char delimiter{','}; ///< Field delimiter.
    bool has_header{true}; ///< First non-empty line contains column names.
};

/**
 * Thrown on any CSV parsing or lookup failure. Message includes the file
 * path (if available), 1-based line number (0 if not applicable), and the
 * offending column name (empty if not applicable).
 */
class CsvParseError : public std::runtime_error {
public:
    CsvParseError(std::string message,
                  std::string source,
                  std::size_t line,
                  std::string column);

    [[nodiscard]] const std::string& source() const noexcept { return source_; }
    [[nodiscard]] std::size_t        line()   const noexcept { return line_; }
    [[nodiscard]] const std::string& column() const noexcept { return column_; }

private:
    std::string source_;
    std::size_t line_;
    std::string column_;
};

class CsvTable;

/**
 * Non-owning view of a single row within a `CsvTable`. Cheap to copy, must
 * not outlive the parent `CsvTable`.
 */
class CsvRow {
public:
    /** Raw field text (empty string if the field is empty). Throws if the column is unknown. */
    [[nodiscard]] std::string_view field(std::string_view column) const;

    /** True if the column exists in the header and the field is non-empty. */
    [[nodiscard]] bool has(std::string_view column) const noexcept;

    /** Parse the field as `double`. Throws `CsvParseError` on failure. */
    [[nodiscard]] double as_double(std::string_view column) const;

    /** Parse the field as `long long`. Throws `CsvParseError` on failure. */
    [[nodiscard]] long long as_int(std::string_view column) const;

    /**
     * Parse the field as a boolean. Accepts (case-insensitive):
     * true/false, 1/0, yes/no. Throws `CsvParseError` on anything else.
     */
    [[nodiscard]] bool as_bool(std::string_view column) const;

    /** 1-based line number in the source file. Useful for error messages. */
    [[nodiscard]] std::size_t line_number() const noexcept { return line_; }

private:
    friend class CsvTable;
    CsvRow(const CsvTable* table, std::size_t row_index, std::size_t line_number);

    const CsvTable* table_;
    std::size_t     row_index_;
    std::size_t     line_;
};

/**
 * A fully parsed CSV table held in memory.
 */
class CsvTable {
public:
    /**
     * Read and parse a CSV file. `path` is used only for error messages;
     * throws `CsvParseError` if the file cannot be opened or parsed.
     */
    [[nodiscard]] static CsvTable read_file(const std::filesystem::path& path,
                                            CsvOptions options = {});

    /** Parse an in-memory CSV string. Handy for tests and pipes. */
    [[nodiscard]] static CsvTable parse_string(std::string_view text,
                                               CsvOptions options = {},
                                               std::string source = "<memory>");

    [[nodiscard]] const std::vector<std::string>& columns() const noexcept { return columns_; }
    [[nodiscard]] std::size_t num_rows() const noexcept { return rows_.size(); }
    [[nodiscard]] bool empty() const noexcept { return rows_.empty(); }
    [[nodiscard]] const std::string& source() const noexcept { return source_; }

    /** True if `column` is present in the header. */
    [[nodiscard]] bool has_column(std::string_view column) const noexcept;

    /** Access a row by index (0-based). Throws `std::out_of_range` if out of bounds. */
    [[nodiscard]] CsvRow row(std::size_t i) const;

    class const_iterator;
    [[nodiscard]] const_iterator begin() const noexcept;
    [[nodiscard]] const_iterator end()   const noexcept;

private:
    friend class CsvRow;
    CsvTable() = default;

    void ingest(std::string_view text, CsvOptions options);

    [[nodiscard]] std::size_t require_column(std::string_view name, std::size_t line) const;
    [[nodiscard]] std::string_view field_at(std::size_t row_index, std::string_view column) const;

    std::vector<std::string>                     columns_;
    std::unordered_map<std::string, std::size_t> column_index_;
    std::vector<std::vector<std::string>>        rows_;
    std::vector<std::size_t>                     row_lines_; ///< source line number of each row
    std::string                                  source_{"<memory>"};
};

class CsvTable::const_iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = CsvRow;
    using difference_type   = std::ptrdiff_t;
    using reference         = CsvRow;
    using pointer           = void;

    const_iterator() = default;
    const_iterator(const CsvTable* table, std::size_t index) : table_(table), index_(index) {}

    [[nodiscard]] CsvRow operator*() const { return table_->row(index_); }
    const_iterator& operator++() { ++index_; return *this; }
    const_iterator  operator++(int) { auto tmp = *this; ++index_; return tmp; }
    [[nodiscard]] bool operator==(const const_iterator& o) const noexcept { return table_ == o.table_ && index_ == o.index_; }
    [[nodiscard]] bool operator!=(const const_iterator& o) const noexcept { return !(*this == o); }

private:
    const CsvTable* table_{nullptr};
    std::size_t     index_{0};
};

inline CsvTable::const_iterator CsvTable::begin() const noexcept { return {this, 0}; }
inline CsvTable::const_iterator CsvTable::end()   const noexcept { return {this, rows_.size()}; }

} // namespace ore::io
