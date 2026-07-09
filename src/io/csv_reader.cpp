#include <ore/io/csv_reader.hpp>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <sstream>
#include <system_error>

namespace ore::io {

namespace {

std::string format_message(std::string_view msg,
                           std::string_view source,
                           std::size_t line,
                           std::string_view column) {
    std::ostringstream oss;
    oss << msg;
    if (!source.empty()) {
        oss << " [source=" << source << "]";
    }
    if (line != 0) {
        oss << " [line=" << line << "]";
    }
    if (!column.empty()) {
        oss << " [column=" << column << "]";
    }
    return oss.str();
}

[[noreturn]] void raise(std::string message,
                        std::string source,
                        std::size_t line,
                        std::string column) {
    throw CsvParseError(std::move(message), std::move(source), line, std::move(column));
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// Parse a single CSV logical line into fields, respecting quoted-field
// escapes. `line` is expected to have any trailing '\r' or '\n' already
// stripped.
std::vector<std::string> split_line(std::string_view line,
                                    char delimiter,
                                    std::size_t line_number,
                                    std::string_view source) {
    std::vector<std::string> fields;
    std::string current;
    current.reserve(32);

    bool in_quotes = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                // Look ahead: doubled `""` means literal quote.
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    current.push_back('"');
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                current.push_back(c);
            }
        } else {
            if (c == '"') {
                if (!current.empty()) {
                    raise(format_message("Unexpected '\"' in the middle of an unquoted field",
                                         source, line_number, ""),
                          std::string(source), line_number, {});
                }
                in_quotes = true;
            } else if (c == delimiter) {
                fields.push_back(std::move(current));
                current.clear();
            } else {
                current.push_back(c);
            }
        }
    }
    if (in_quotes) {
        raise(format_message("Unterminated quoted field (embedded newlines are not supported)",
                             source, line_number, ""),
              std::string(source), line_number, {});
    }
    fields.push_back(std::move(current));
    return fields;
}

std::string trim(std::string_view s) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    auto first = std::find_if(s.begin(), s.end(), not_space);
    auto last  = std::find_if(s.rbegin(), s.rend(), not_space).base();
    if (first >= last) return {};
    return std::string(first, last);
}

} // namespace

// -----------------------------------------------------------------------------
// CsvParseError
// -----------------------------------------------------------------------------
CsvParseError::CsvParseError(std::string message,
                             std::string source,
                             std::size_t line,
                             std::string column)
    : std::runtime_error(std::move(message)),
      source_(std::move(source)),
      line_(line),
      column_(std::move(column)) {}

// -----------------------------------------------------------------------------
// CsvRow
// -----------------------------------------------------------------------------
CsvRow::CsvRow(const CsvTable* table, std::size_t row_index, std::size_t line_number)
    : table_(table), row_index_(row_index), line_(line_number) {}

std::string_view CsvRow::field(std::string_view column) const {
    return table_->field_at(row_index_, column);
}

bool CsvRow::has(std::string_view column) const noexcept {
    if (!table_->has_column(column)) return false;
    const auto& val = table_->field_at(row_index_, column);
    return !val.empty();
}

double CsvRow::as_double(std::string_view column) const {
    const std::string_view raw = field(column);
    if (raw.empty()) {
        raise(format_message("Empty field where a number was expected",
                             table_->source(), line_, column),
              table_->source(), line_, std::string(column));
    }
    // std::from_chars for double is C++17 in the standard but MSVC & recent
    // GCC/Clang all support it. Handles leading whitespace via manual trim.
    std::string trimmed = trim(raw);
    double value{};
    const auto* first = trimmed.data();
    const auto* last  = first + trimmed.size();
    const auto result = std::from_chars(first, last, value);
    if (result.ec != std::errc{} || result.ptr != last) {
        raise(format_message("Could not parse field as a floating-point number",
                             table_->source(), line_, column),
              table_->source(), line_, std::string(column));
    }
    return value;
}

long long CsvRow::as_int(std::string_view column) const {
    const std::string_view raw = field(column);
    if (raw.empty()) {
        raise(format_message("Empty field where an integer was expected",
                             table_->source(), line_, column),
              table_->source(), line_, std::string(column));
    }
    std::string trimmed = trim(raw);
    long long value{};
    const auto* first = trimmed.data();
    const auto* last  = first + trimmed.size();
    const auto result = std::from_chars(first, last, value);
    if (result.ec != std::errc{} || result.ptr != last) {
        raise(format_message("Could not parse field as an integer",
                             table_->source(), line_, column),
              table_->source(), line_, std::string(column));
    }
    return value;
}

bool CsvRow::as_bool(std::string_view column) const {
    const std::string_view raw = field(column);
    if (raw.empty()) {
        raise(format_message("Empty field where a boolean was expected",
                             table_->source(), line_, column),
              table_->source(), line_, std::string(column));
    }
    std::string t = to_lower(trim(raw));
    if (t == "true" || t == "1" || t == "yes")  return true;
    if (t == "false" || t == "0" || t == "no")  return false;
    raise(format_message("Could not parse field as a boolean",
                         table_->source(), line_, column),
          table_->source(), line_, std::string(column));
}

// -----------------------------------------------------------------------------
// CsvTable
// -----------------------------------------------------------------------------
CsvTable CsvTable::read_file(const std::filesystem::path& path, CsvOptions options) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        raise("Could not open CSV file for reading", path.string(), 0, "");
    }
    std::ostringstream buf;
    buf << in.rdbuf();
    CsvTable table = parse_string(buf.str(), options, path.string());
    return table;
}

CsvTable CsvTable::parse_string(std::string_view text, CsvOptions options, std::string source) {
    CsvTable t;
    t.source_ = std::move(source);
    t.ingest(text, options);
    return t;
}

bool CsvTable::has_column(std::string_view column) const noexcept {
    return column_index_.find(std::string(column)) != column_index_.end();
}

CsvRow CsvTable::row(std::size_t i) const {
    if (i >= rows_.size()) {
        throw std::out_of_range("CsvTable::row index out of range");
    }
    return CsvRow{this, i, row_lines_[i]};
}

std::size_t CsvTable::require_column(std::string_view name, std::size_t line) const {
    auto it = column_index_.find(std::string(name));
    if (it == column_index_.end()) {
        raise(format_message("Unknown column", source_, line, name),
              source_, line, std::string(name));
    }
    return it->second;
}

std::string_view CsvTable::field_at(std::size_t row_index, std::string_view column) const {
    const std::size_t col = require_column(column, row_lines_[row_index]);
    const auto& r = rows_[row_index];
    if (col >= r.size()) {
        return {};
    }
    return r[col];
}

void CsvTable::ingest(std::string_view text, CsvOptions options) {
    // Strip a leading UTF-8 BOM if present. Some CSV producers (notably
    // Excel) emit one; without this, the first column name would be
    // silently corrupted.
    if (text.size() >= 3 &&
        static_cast<unsigned char>(text[0]) == 0xEF &&
        static_cast<unsigned char>(text[1]) == 0xBB &&
        static_cast<unsigned char>(text[2]) == 0xBF) {
        text.remove_prefix(3);
    }

    std::size_t line_number = 0;
    std::size_t start = 0;
    bool header_seen = !options.has_header; // if no header expected, we're already past it

    const auto process_line = [&](std::string_view raw_line) {
        ++line_number;
        // Strip a trailing '\r' if present (handle \r\n).
        if (!raw_line.empty() && raw_line.back() == '\r') {
            raw_line.remove_suffix(1);
        }
        // Skip entirely blank lines.
        if (raw_line.empty()) {
            return;
        }
        auto fields = split_line(raw_line, options.delimiter, line_number, source_);
        if (!header_seen) {
            columns_.reserve(fields.size());
            for (std::size_t i = 0; i < fields.size(); ++i) {
                std::string name = trim(fields[i]);
                if (name.empty()) {
                    raise(format_message("Blank column name in header",
                                         source_, line_number, ""),
                          source_, line_number, {});
                }
                if (!column_index_.emplace(name, i).second) {
                    raise(format_message("Duplicate column name in header: " + name,
                                         source_, line_number, name),
                          source_, line_number, name);
                }
                columns_.push_back(std::move(name));
            }
            header_seen = true;
            return;
        }
        // Enforce field count. Extra fields are treated as an error under
        // strict parsing; missing fields are padded with empty strings so
        // callers can decide per-column whether emptiness is fatal.
        if (fields.size() > columns_.size() && !columns_.empty()) {
            raise(format_message("Row has more fields than the header",
                                 source_, line_number, ""),
                  source_, line_number, {});
        }
        if (!columns_.empty() && fields.size() < columns_.size()) {
            fields.resize(columns_.size());
        }
        rows_.push_back(std::move(fields));
        row_lines_.push_back(line_number);
    };

    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\n') {
            process_line(std::string_view(text.data() + start, i - start));
            start = i + 1;
        }
    }
    if (start < text.size()) {
        process_line(std::string_view(text.data() + start, text.size() - start));
    }

    if (options.has_header && !header_seen) {
        raise("CSV is empty; expected a header row", source_, 0, "");
    }
}

} // namespace ore::io
