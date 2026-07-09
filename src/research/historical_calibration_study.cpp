#include <ore/research/historical_calibration_study.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <ore/analytics/option_chain_calibrator.hpp>
#include <ore/analytics/volatility_analytics.hpp>
#include <ore/core/market_snapshot.hpp>
#include <ore/core/option.hpp>
#include <ore/core/option_chain.hpp>
#include <ore/core/option_market_snapshot.hpp>
#include <ore/core/types.hpp>
#include <ore/numerics/solver_result.hpp>
#include <ore/research/research_context.hpp>
#include <ore/research/research_csv.hpp>
#include <ore/research/research_report.hpp>
#include <ore/research/research_study.hpp>

namespace ore::research {

namespace {

using ore::analytics::CalibrationReport;
using ore::analytics::CalibrationResult;
using ore::analytics::OptionChainCalibrator;
using ore::analytics::SkewMetrics;
using ore::analytics::SkipReason;
using ore::analytics::TermStructure;
using ore::analytics::VolatilitySmile;
using ore::analytics::VolatilitySurface;

/// ISO-8601 date formatter, sharing the `ResearchCsvWriter` conventions.
/// Retained as a convenience alias even though callers currently reach
/// for `ResearchCsvWriter::date` directly; `[[maybe_unused]]` keeps the
/// `-Werror=unused-function` build clean without deleting it.
[[maybe_unused, nodiscard]] std::string fmt_date(std::chrono::year_month_day d) {
    return ResearchCsvWriter::date(d);
}

/// Optional formatter that treats a non-finite double as absent —
/// necessary for `VolatilitySurface` (missing cells are `NaN` by
/// convention) and `TermStructure` (missing brackets return `NaN`).
[[nodiscard]] std::optional<double> finite_or_nullopt(double x) noexcept {
    return std::isfinite(x) ? std::optional<double>{x} : std::nullopt;
}

/// Human-readable serialisation of `Moneyness` for the smile CSV.
/// Reuses `ore::analytics::to_string` verbatim so the strings never
/// drift from the smile-header names used by `plot_smile.py`.
[[nodiscard]] std::string moneyness_name(ore::analytics::Moneyness m) {
    return std::string(ore::analytics::to_string(m));
}

/// Read a `CalibrationResult` into the compact `CalibrationRow` used
/// by the historical CSV. Keeps only the fields the CSV writes —
/// `contract_symbol` is dropped to keep memory bounded across
/// 10-year archives (a 4-column identifier suffices for research
/// aggregation).
[[nodiscard]] HistoricalCalibrationStudy::CalibrationRow
to_calibration_row(std::chrono::year_month_day date,
                   const CalibrationResult& r)
{
    HistoricalCalibrationStudy::CalibrationRow row{};
    row.date            = date;
    row.expiration      = r.option.expiration;
    row.strike          = r.option.strike;
    row.type            = r.option.type;
    row.bid             = r.bid;
    row.ask             = r.ask;
    row.last            = r.last;
    row.mid_price       = r.mid_price;
    row.provider_iv     = r.provider_iv;
    row.computed_iv     = r.computed_iv;
    row.absolute_error  = r.absolute_error;
    row.relative_error  = r.relative_error;
    row.solver_status   = r.solver_status;
    row.iterations      = r.iterations;
    row.used_bisection  = r.used_bisection;
    row.solver_residual = r.solver_residual;
    row.skip_reason     = r.skip_reason;
    return row;
}

} // namespace

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void HistoricalCalibrationStudy::begin(const ResearchContext& /*ctx*/) {
    // Clear all accumulators. `begin()` is called on the primary
    // study once; clones are started fresh via `clone()` and are
    // therefore already empty.
    calibration_rows_.clear();
    smile_rows_.clear();
    term_rows_.clear();
    surface_rows_.clear();
    skew_rows_.clear();
    stats_                = Statistics{};
    total_contracts_      = 0;
    total_skipped_        = 0;
    total_failed_solves_  = 0;
    total_iv_comparisons_ = 0;
    sum_abs_iv_error_     = 0.0;
    sum_convergence_rate_ = 0.0;
    days_with_attempts_   = 0;
}

void HistoricalCalibrationStudy::process(const ResearchContext& ctx) {
    // 1) Calibrate today's chain -----------------------------------------
    const auto date      = ctx.date();
    const auto& chain    = ctx.chain();
    const auto& market   = ctx.market();

    OptionChainCalibrator calibrator{config_.calibrator};
    const CalibrationReport report = calibrator.calibrate(chain);

    // 2) Build the volatility structures over this day's report ---------
    // Every builder returns an empty structure when the report itself
    // is empty (no calibrated contracts) — no defensive branching
    // needed here. Skew is optional because it requires a smile.
    std::vector<VolatilitySmile> smiles;
    TermStructure                term;
    VolatilitySurface            surface;
    std::vector<SkewMetrics>     skews;

    if (config_.export_smiles || config_.export_skew) {
        smiles = ore::analytics::build_smiles(
            report, market, config_.moneyness_convention);
    }
    if (config_.export_term_structure) {
        term = ore::analytics::build_term_structure(report, market);
    }
    if (config_.export_surface) {
        surface = ore::analytics::build_surface(report, market);
    }
    if (config_.export_skew && !smiles.empty()) {
        skews = ore::analytics::compute_skew_metrics(smiles, market);
    }

    // 3) Extract compact rows into thread-local scratch buffers ---------
    // Two reasons for the scratch buffers:
    //   - keep the study's mutex locked only during the final splice;
    //   - avoid reallocating the primary vectors on every day.
    std::vector<CalibrationRow>    calib_rows;
    std::vector<SmileRow>          smile_rows;
    std::vector<TermStructureRow>  term_rows_local;
    std::vector<SurfaceRow>        surface_rows_local;
    std::vector<SkewRow>           skew_rows_local;

    std::size_t day_contracts       = 0;
    std::size_t day_skipped         = 0;
    std::size_t day_failed          = 0;
    std::size_t day_comparisons     = 0;
    std::size_t day_attempts        = 0;
    std::size_t day_converged       = 0;
    double      day_abs_error_sum   = 0.0;

    if (config_.export_calibration) {
        calib_rows.reserve(report.results.size());
    }

    for (const auto& r : report.results) {
        ++day_contracts;

        if (r.was_skipped()) {
            ++day_skipped;
        } else {
            ++day_attempts;
            if (r.was_calibrated()) {
                ++day_converged;
            } else {
                ++day_failed;
            }
        }
        if (r.absolute_error.has_value()) {
            ++day_comparisons;
            day_abs_error_sum += *r.absolute_error;
        }

        if (config_.export_calibration) {
            const bool keep = config_.include_skipped_in_calibration
                              || r.was_calibrated();
            if (keep) {
                calib_rows.push_back(to_calibration_row(date, r));
            }
        }
    }

    // Smile rows: iterate every point of every smile.
    if (config_.export_smiles) {
        std::size_t smile_count = 0;
        for (const auto& s : smiles) {
            smile_count += s.size();
        }
        smile_rows.reserve(smile_count);
        for (const auto& s : smiles) {
            const auto n = s.size();
            for (std::size_t i = 0; i < n; ++i) {
                SmileRow row{};
                row.date                 = date;
                row.expiration           = s.expiration;
                row.time_to_expiry       = s.time_to_expiry;
                row.strike               = s.strikes[i];
                row.type                 = s.types[i];
                row.moneyness_convention = s.moneyness_convention;
                row.moneyness            = s.moneyness[i];
                row.implied_volatility   = s.implied_volatility[i];
                smile_rows.push_back(row);
            }
        }
    }

    // Term-structure rows: one per expiration, with `NaN` mapped to
    // `nullopt` so the CSV writer emits an empty field.
    if (config_.export_term_structure) {
        const auto n = term.size();
        term_rows_local.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            TermStructureRow row{};
            row.date            = date;
            row.expiration      = term.expirations[i];
            row.time_to_expiry  = term.maturities[i];
            row.atm_iv          = finite_or_nullopt(term.atm_iv[i]);
            term_rows_local.push_back(row);
        }
    }

    // Surface rows: pivot the 2D grid to long form so pandas can
    // pivot back with a single call.
    if (config_.export_surface) {
        const auto rows = surface.rows();
        const auto cols = surface.cols();
        surface_rows_local.reserve(rows * cols);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                SurfaceRow row{};
                row.date               = date;
                row.expiration         = surface.expirations[i];
                row.time_to_expiry     = surface.maturities[i];
                row.strike             = surface.strikes[j];
                row.implied_volatility =
                    finite_or_nullopt(surface.implied_vols[i][j]);
                surface_rows_local.push_back(row);
            }
        }
    }

    // Skew rows: one per skew, all fields already optional.
    if (config_.export_skew) {
        skew_rows_local.reserve(skews.size());
        for (const auto& s : skews) {
            SkewRow row{};
            row.date            = date;
            row.expiration      = s.expiration;
            row.time_to_expiry  = s.time_to_expiry;
            row.atm_iv          = s.atm_iv;
            row.call_25delta_iv = s.call_25delta_iv;
            row.put_25delta_iv  = s.put_25delta_iv;
            row.risk_reversal   = s.risk_reversal;
            row.butterfly       = s.butterfly;
            skew_rows_local.push_back(row);
        }
    }

    // 4) Splice into the study's accumulators under the study mutex.
    //    Locks are only held for the O(N) insert; no analytics work
    //    runs inside the critical section.
    std::lock_guard<std::mutex> lock(mutex_);
    calibration_rows_.insert(calibration_rows_.end(),
        std::make_move_iterator(calib_rows.begin()),
        std::make_move_iterator(calib_rows.end()));
    smile_rows_.insert(smile_rows_.end(),
        std::make_move_iterator(smile_rows.begin()),
        std::make_move_iterator(smile_rows.end()));
    term_rows_.insert(term_rows_.end(),
        std::make_move_iterator(term_rows_local.begin()),
        std::make_move_iterator(term_rows_local.end()));
    surface_rows_.insert(surface_rows_.end(),
        std::make_move_iterator(surface_rows_local.begin()),
        std::make_move_iterator(surface_rows_local.end()));
    skew_rows_.insert(skew_rows_.end(),
        std::make_move_iterator(skew_rows_local.begin()),
        std::make_move_iterator(skew_rows_local.end()));

    total_contracts_       += day_contracts;
    total_skipped_         += day_skipped;
    total_failed_solves_   += day_failed;
    total_iv_comparisons_  += day_comparisons;
    sum_abs_iv_error_      += day_abs_error_sum;
    if (day_attempts > 0) {
        ++days_with_attempts_;
        sum_convergence_rate_ +=
            static_cast<double>(day_converged) / static_cast<double>(day_attempts);
    }
}

// ---------------------------------------------------------------------------
// CSV writers
// ---------------------------------------------------------------------------

namespace {

/// Convert a `SkipReason` to its short human name. We reuse the
/// existing string-conversion helper so the strings never drift.
[[nodiscard]] std::string skip_reason_name(SkipReason r) {
    return std::string(ore::analytics::to_string(r));
}

/// Convert a `SolverStatus` to its short human name. Delegates to the
/// numerics-module helper.
[[nodiscard]] std::string solver_status_name(ore::numerics::SolverStatus s) {
    return std::string(ore::numerics::to_string(s));
}

void write_calibration_csv(
    const std::filesystem::path& path,
    const std::vector<HistoricalCalibrationStudy::CalibrationRow>& rows)
{
    ResearchCsvWriter w(path, {
        "date",
        "expiration",
        "strike",
        "option_type",
        "bid",
        "ask",
        "last",
        "mid_price",
        "provider_iv",
        "computed_iv",
        "absolute_error",
        "relative_error",
        "solver_status",
        "iterations",
        "used_bisection",
        "solver_residual",
        "skip_reason",
    });
    for (const auto& r : rows) {
        w.write_row({
            ResearchCsvWriter::date(r.date),
            ResearchCsvWriter::date(r.expiration),
            ResearchCsvWriter::number(r.strike),
            std::string(ore::core::to_string(r.type)),
            ResearchCsvWriter::number(r.bid),
            ResearchCsvWriter::number(r.ask),
            ResearchCsvWriter::number(r.last),
            ResearchCsvWriter::number(r.mid_price),
            ResearchCsvWriter::optional_number(r.provider_iv),
            ResearchCsvWriter::optional_number(r.computed_iv),
            ResearchCsvWriter::optional_number(r.absolute_error),
            ResearchCsvWriter::optional_number(r.relative_error),
            solver_status_name(r.solver_status),
            ResearchCsvWriter::integer(r.iterations),
            r.used_bisection ? std::string("1") : std::string("0"),
            ResearchCsvWriter::number(r.solver_residual),
            skip_reason_name(r.skip_reason),
        });
    }
}

void write_smiles_csv(
    const std::filesystem::path& path,
    const std::vector<HistoricalCalibrationStudy::SmileRow>& rows)
{
    ResearchCsvWriter w(path, {
        "date",
        "expiration",
        "time_to_expiry",
        "strike",
        "option_type",
        "moneyness_convention",
        "moneyness",
        "implied_volatility",
    });
    for (const auto& r : rows) {
        w.write_row({
            ResearchCsvWriter::date(r.date),
            ResearchCsvWriter::date(r.expiration),
            ResearchCsvWriter::number(r.time_to_expiry),
            ResearchCsvWriter::number(r.strike),
            std::string(ore::core::to_string(r.type)),
            moneyness_name(r.moneyness_convention),
            ResearchCsvWriter::number(r.moneyness),
            ResearchCsvWriter::number(r.implied_volatility),
        });
    }
}

void write_term_structure_csv(
    const std::filesystem::path& path,
    const std::vector<HistoricalCalibrationStudy::TermStructureRow>& rows)
{
    ResearchCsvWriter w(path, {
        "date",
        "expiration",
        "time_to_expiry",
        "atm_iv",
    });
    for (const auto& r : rows) {
        w.write_row({
            ResearchCsvWriter::date(r.date),
            ResearchCsvWriter::date(r.expiration),
            ResearchCsvWriter::number(r.time_to_expiry),
            ResearchCsvWriter::optional_number(r.atm_iv),
        });
    }
}

void write_surface_csv(
    const std::filesystem::path& path,
    const std::vector<HistoricalCalibrationStudy::SurfaceRow>& rows)
{
    ResearchCsvWriter w(path, {
        "date",
        "expiration",
        "time_to_expiry",
        "strike",
        "implied_volatility",
    });
    for (const auto& r : rows) {
        w.write_row({
            ResearchCsvWriter::date(r.date),
            ResearchCsvWriter::date(r.expiration),
            ResearchCsvWriter::number(r.time_to_expiry),
            ResearchCsvWriter::number(r.strike),
            ResearchCsvWriter::optional_number(r.implied_volatility),
        });
    }
}

void write_skew_csv(
    const std::filesystem::path& path,
    const std::vector<HistoricalCalibrationStudy::SkewRow>& rows)
{
    ResearchCsvWriter w(path, {
        "date",
        "expiration",
        "time_to_expiry",
        "atm_iv",
        "call_25delta_iv",
        "put_25delta_iv",
        "risk_reversal",
        "butterfly",
    });
    for (const auto& r : rows) {
        w.write_row({
            ResearchCsvWriter::date(r.date),
            ResearchCsvWriter::date(r.expiration),
            ResearchCsvWriter::number(r.time_to_expiry),
            ResearchCsvWriter::optional_number(r.atm_iv),
            ResearchCsvWriter::optional_number(r.call_25delta_iv),
            ResearchCsvWriter::optional_number(r.put_25delta_iv),
            ResearchCsvWriter::optional_number(r.risk_reversal),
            ResearchCsvWriter::optional_number(r.butterfly),
        });
    }
}

} // namespace

void HistoricalCalibrationStudy::end(const ResearchContext& ctx,
                                     ResearchReport& report)
{
    // Defensive sort by (date, expiration, strike) so parallel merges
    // that happened out-of-order end up deterministic on disk. The
    // engine already guarantees date order when merges are performed
    // in worker order, but a stable sort here means "study output is
    // ordered" is a property of the study, not of the engine.
    auto by_calibration = [](const CalibrationRow& a, const CalibrationRow& b) {
        if (a.date != b.date)             return a.date < b.date;
        if (a.expiration != b.expiration) return a.expiration < b.expiration;
        if (a.strike != b.strike)         return a.strike < b.strike;
        return static_cast<int>(a.type) < static_cast<int>(b.type);
    };
    auto by_smile = [](const SmileRow& a, const SmileRow& b) {
        if (a.date != b.date)             return a.date < b.date;
        if (a.expiration != b.expiration) return a.expiration < b.expiration;
        if (a.strike != b.strike)         return a.strike < b.strike;
        return static_cast<int>(a.type) < static_cast<int>(b.type);
    };
    auto by_term = [](const TermStructureRow& a, const TermStructureRow& b) {
        if (a.date != b.date) return a.date < b.date;
        return a.expiration < b.expiration;
    };
    auto by_surface = [](const SurfaceRow& a, const SurfaceRow& b) {
        if (a.date != b.date)             return a.date < b.date;
        if (a.expiration != b.expiration) return a.expiration < b.expiration;
        return a.strike < b.strike;
    };
    auto by_skew = [](const SkewRow& a, const SkewRow& b) {
        if (a.date != b.date) return a.date < b.date;
        return a.expiration < b.expiration;
    };

    std::stable_sort(calibration_rows_.begin(), calibration_rows_.end(), by_calibration);
    std::stable_sort(smile_rows_.begin(),       smile_rows_.end(),       by_smile);
    std::stable_sort(term_rows_.begin(),        term_rows_.end(),        by_term);
    std::stable_sort(surface_rows_.begin(),     surface_rows_.end(),     by_surface);
    std::stable_sort(skew_rows_.begin(),        skew_rows_.end(),        by_skew);

    // Recompute aggregate statistics from the row vectors. This is
    // idempotent — parallel merges have already accumulated the raw
    // counts through `total_*_` running fields; end() finalises the
    // derived means from those counts.
    stats_.days_processed        = report.processed_days;
    stats_.total_contracts       = total_contracts_;
    stats_.total_skipped         = total_skipped_;
    stats_.total_failed_solves   = total_failed_solves_;
    stats_.total_iv_comparisons  = total_iv_comparisons_;
    stats_.total_calibrated      =
        total_contracts_ - total_skipped_ - total_failed_solves_;
    stats_.mean_absolute_iv_error =
        (total_iv_comparisons_ == 0) ? 0.0 :
        (sum_abs_iv_error_ / static_cast<double>(total_iv_comparisons_));
    stats_.mean_convergence_rate =
        (days_with_attempts_ == 0) ? 0.0 :
        (sum_convergence_rate_ / static_cast<double>(days_with_attempts_));

    report.processed_contracts = stats_.total_calibrated;
    report.skipped_contracts   = total_skipped_ + total_failed_solves_;

    // Write the five CSVs. Every path is anchored at
    // `ctx.output_dir()`; no filesystem path is ever hardcoded.
    const auto& out_dir = ctx.output_dir();

    if (config_.export_calibration) {
        const auto path = out_dir / config_.calibration_filename;
        write_calibration_csv(path, calibration_rows_);
        report.generated_files.push_back(path);
    }
    if (config_.export_smiles) {
        const auto path = out_dir / config_.smiles_filename;
        write_smiles_csv(path, smile_rows_);
        report.generated_files.push_back(path);
    }
    if (config_.export_term_structure) {
        const auto path = out_dir / config_.term_structure_filename;
        write_term_structure_csv(path, term_rows_);
        report.generated_files.push_back(path);
    }
    if (config_.export_surface) {
        const auto path = out_dir / config_.surface_filename;
        write_surface_csv(path, surface_rows_);
        report.generated_files.push_back(path);
    }
    if (config_.export_skew) {
        const auto path = out_dir / config_.skew_filename;
        write_skew_csv(path, skew_rows_);
        report.generated_files.push_back(path);
    }
}

// ---------------------------------------------------------------------------
// Parallel-execution hooks
// ---------------------------------------------------------------------------

std::unique_ptr<ResearchStudy> HistoricalCalibrationStudy::clone() const {
    // A fresh study with the same configuration and empty
    // accumulators. Not copying the row vectors is critical —
    // workers must start empty or the merge would double-count.
    return std::make_unique<HistoricalCalibrationStudy>(config_);
}

void HistoricalCalibrationStudy::merge(const ResearchStudy& other) {
    // Silently drop mis-typed merges; a mismatch is a programmer
    // error and aborting a long-running research job at merge time
    // would waste an entire compute run.
    const auto* rhs = dynamic_cast<const HistoricalCalibrationStudy*>(&other);
    if (!rhs) return;

    std::lock_guard<std::mutex> lock(mutex_);
    calibration_rows_.insert(calibration_rows_.end(),
        rhs->calibration_rows_.begin(), rhs->calibration_rows_.end());
    smile_rows_.insert(smile_rows_.end(),
        rhs->smile_rows_.begin(), rhs->smile_rows_.end());
    term_rows_.insert(term_rows_.end(),
        rhs->term_rows_.begin(), rhs->term_rows_.end());
    surface_rows_.insert(surface_rows_.end(),
        rhs->surface_rows_.begin(), rhs->surface_rows_.end());
    skew_rows_.insert(skew_rows_.end(),
        rhs->skew_rows_.begin(), rhs->skew_rows_.end());

    total_contracts_       += rhs->total_contracts_;
    total_skipped_         += rhs->total_skipped_;
    total_failed_solves_   += rhs->total_failed_solves_;
    total_iv_comparisons_  += rhs->total_iv_comparisons_;
    sum_abs_iv_error_      += rhs->sum_abs_iv_error_;
    sum_convergence_rate_  += rhs->sum_convergence_rate_;
    days_with_attempts_    += rhs->days_with_attempts_;
}

} // namespace ore::research
