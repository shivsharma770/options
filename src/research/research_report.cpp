#include <ore/research/research_report.hpp>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <ostream>
#include <string>

namespace ore::research {

void ResearchReport::print(std::ostream& os) const {
    // Legend layout: label column wide enough for the longest field
    // ("Skipped contracts") and one blank column of separation. The
    // values are left-aligned because most of them are integers of
    // very different widths — right-aligning would look uneven.
    os << "Study            : " << study_name          << '\n';
    os << "Processed days   : " << processed_days      << '\n';
    os << "Processed contr. : " << processed_contracts << '\n';
    os << "Skipped contr.   : " << skipped_contracts   << '\n';
    os << "Runtime (s)      : "
       << std::fixed << std::setprecision(3)
       << runtime_seconds
       << std::defaultfloat << '\n';

    if (!generated_files.empty()) {
        os << "Generated files  : " << generated_files.size() << '\n';
        for (const auto& p : generated_files) {
            os << "  - " << p.string() << '\n';
        }
    } else {
        os << "Generated files  : 0\n";
    }

    // Warnings / errors: cap the printed detail so the summary stays
    // readable even when a study runs on years of data. The full
    // vectors remain available on the returned struct for callers
    // that need every entry.
    constexpr std::size_t kMaxPrinted = 10;

    if (!warnings.empty()) {
        os << "Warnings         : " << warnings.size();
        os << " (first " << std::min(warnings.size(), kMaxPrinted) << ")\n";
        for (std::size_t i = 0; i < std::min(warnings.size(), kMaxPrinted); ++i) {
            os << "  W: " << warnings[i] << '\n';
        }
    } else {
        os << "Warnings         : 0\n";
    }

    if (!errors.empty()) {
        os << "Errors           : " << errors.size();
        os << " (first " << std::min(errors.size(), kMaxPrinted) << ")\n";
        for (std::size_t i = 0; i < std::min(errors.size(), kMaxPrinted); ++i) {
            os << "  E: " << errors[i] << '\n';
        }
    } else {
        os << "Errors           : 0\n";
    }
}

} // namespace ore::research
