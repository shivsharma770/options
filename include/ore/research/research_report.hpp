/**
 * @file research_report.hpp
 * @brief Summary of a single study run.
 *
 * `ResearchReport` is what `HistoricalResearchEngine::run(...)`
 * returns. It captures three kinds of information:
 *
 *   1. **Progress counters** — how many trading days and contracts
 *      were seen, how many contracts a study chose to skip (deep-OTM,
 *      zero-quote, missing IV, ...).
 *   2. **Timing** — wall-clock runtime.
 *   3. **Provenance** — the CSV files the study wrote, warnings the
 *      study surfaced, and any errors caught by the engine while
 *      dispatching work.
 *
 * The struct is intentionally a plain aggregate. Consumers project it
 * into whatever downstream format they need (stdout summary,
 * JSON-lines for CI, human-readable Markdown).
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

namespace ore::research {

/**
 * @struct ResearchReport
 * @brief Result summary returned by `HistoricalResearchEngine::run`.
 */
struct ResearchReport {
    /** Short study identifier (e.g. `"IVValidationStudy"`). */
    std::string study_name{};

    /** Number of trading days the engine dispatched into the study. */
    std::size_t processed_days{0};

    /**
     * Number of individual contracts the study accepted for
     * processing. `processed_contracts + skipped_contracts` equals
     * the total number of contracts the engine saw, but a study is
     * free to redefine "processed" for its own purposes — see each
     * study's `end()` docstring.
     */
    std::size_t processed_contracts{0};

    /**
     * Number of contracts the study explicitly declined to process
     * (missing IV, zero-quote, expired, malformed, ...). Studies
     * increment this counter as they filter their input.
     */
    std::size_t skipped_contracts{0};

    /** Wall-clock time from engine `run()` entry to exit, in seconds. */
    double runtime_seconds{0.0};

    /** Files the study wrote during `end()`. */
    std::vector<std::filesystem::path> generated_files{};

    /**
     * Non-fatal messages a study emitted (e.g. "IV solver did not
     * converge for 42 rows"). Not the same as `errors`: warnings do
     * not indicate the run itself was compromised.
     */
    std::vector<std::string> warnings{};

    /**
     * Fatal per-day exceptions the engine caught and swallowed to
     * keep the rest of the run going. Empty in a fully-clean run.
     */
    std::vector<std::string> errors{};

    /**
     * Print a compact human-readable summary to `os` — the same
     * format `examples/historical_research.cpp` uses.
     */
    void print(std::ostream& os) const;
};

} // namespace ore::research
