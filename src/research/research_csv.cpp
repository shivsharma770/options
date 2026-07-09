#include <ore/research/research_csv.hpp>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ore::research {

ResearchCsvWriter::ResearchCsvWriter(
    const std::filesystem::path& path,
    std::initializer_list<std::string_view> columns)
    : path_(path)
{
    // Ensure the parent directory exists — studies commonly derive
    // paths like `output_dir / "iv_validation.csv"` and the caller
    // may not have pre-created `output_dir`.
    if (path_.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(path_.parent_path(), ec);
    }
    out_.open(path_, std::ios::binary | std::ios::trunc);
    if (!out_) {
        std::ostringstream oss;
        oss << "ResearchCsvWriter: cannot open '" << path_.string() << "' for writing";
        throw std::runtime_error(oss.str());
    }

    // Emit header. Columns are string_views into either literals or
    // strings the caller owns for the duration of the call — we
    // copy them into the file immediately so no lifetime concerns
    // survive this constructor.
    bool first = true;
    for (const auto& c : columns) {
        if (!first) out_.put(',');
        out_.write(c.data(), static_cast<std::streamsize>(c.size()));
        first = false;
    }
    out_.put('\n');
}

std::string ResearchCsvWriter::number(double x) {
    // %.17g is the classic "double round-trip" format: any IEEE-754
    // double serialised with %.17g re-parses to the same double.
    // std::to_chars with std::chars_format::general | precision 17
    // would be preferable in principle but is not portable across
    // GCC/MSVC/Clang for `double` at C++17 baseline; the project's
    // baseline is C++20 and the situation is better, but snprintf
    // is universally available and identical in output. Buffer
    // sized generously — %.17g never exceeds ~24 chars for finite
    // doubles, but +Inf and NaN are longer.
    char buf[32];
    const int n = std::snprintf(buf, sizeof(buf), "%.17g", x);
    if (n <= 0) return {};
    return std::string(buf, static_cast<std::size_t>(n));
}

std::string ResearchCsvWriter::integer(long long x) {
    char buf[32];
    const int n = std::snprintf(buf, sizeof(buf), "%lld", x);
    if (n <= 0) return {};
    return std::string(buf, static_cast<std::size_t>(n));
}

std::string ResearchCsvWriter::integer(std::size_t x) {
    char buf[32];
    const int n = std::snprintf(buf, sizeof(buf), "%zu", x);
    if (n <= 0) return {};
    return std::string(buf, static_cast<std::size_t>(n));
}

std::string ResearchCsvWriter::date(std::chrono::year_month_day ymd) {
    // Deliberately manual formatting — `std::format` with the
    // %F specifier would work but is not yet available on all
    // supported compilers/CIs (GCC 11 for example). Manual keeps
    // the project compiling with the same minimum C++20 toolchain
    // it uses everywhere else.
    char buf[16];
    const int y = static_cast<int>(ymd.year());
    const unsigned m = static_cast<unsigned>(ymd.month());
    const unsigned d = static_cast<unsigned>(ymd.day());
    const int n = std::snprintf(buf, sizeof(buf), "%04d-%02u-%02u", y, m, d);
    if (n <= 0) return {};
    return std::string(buf, static_cast<std::size_t>(n));
}

std::string ResearchCsvWriter::optional_number(std::optional<double> v) {
    if (!v.has_value()) return {}; // empty field → pandas NaN
    return number(*v);
}

void ResearchCsvWriter::write_row(std::initializer_list<std::string> fields) {
    bool first = true;
    for (const auto& f : fields) {
        if (!first) out_.put(',');
        // No embedded commas or newlines are permitted by contract.
        // Studies only write numbers, dates, and enum names — none
        // of which contain either — so we don't waste cycles on
        // escaping. If a future study needs it, add a `quote(...)`
        // helper here.
        out_.write(f.data(), static_cast<std::streamsize>(f.size()));
        first = false;
    }
    out_.put('\n');
}

} // namespace ore::research
