#include <ore/analytics/volatility_analytics.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <ostream>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <ore/core/types.hpp>
#include <ore/pricing/black_scholes_engine.hpp>

namespace ore::analytics {

namespace {

using ore::core::MarketSnapshot;
using ore::core::OptionType;
using ore::pricing::BlackScholesEngine;

// -----------------------------------------------------------------------------
// Small formatting helpers — shared across every write_csv overload.
// -----------------------------------------------------------------------------

std::string format_double(double x) {
    if (!std::isfinite(x)) return {};
    char buf[32];
    const int n = std::snprintf(buf, sizeof(buf), "%.17g", x);
    return (n > 0) ? std::string(buf, buf + n) : std::string{};
}

std::string format_optional(const std::optional<double>& v) {
    return v.has_value() ? format_double(*v) : std::string{};
}

std::string format_date(const std::chrono::year_month_day& ymd) {
    char buf[16];
    const int y = static_cast<int>(ymd.year());
    const unsigned m = static_cast<unsigned>(ymd.month());
    const unsigned d = static_cast<unsigned>(ymd.day());
    const int n = std::snprintf(buf, sizeof(buf), "%04d-%02u-%02u", y, m, d);
    return (n > 0) ? std::string(buf, buf + n) : std::string{};
}

constexpr std::string_view type_string(OptionType t) noexcept {
    return t == OptionType::Call ? "call" : "put";
}

double year_fraction(const std::chrono::year_month_day& valuation,
                     const std::chrono::year_month_day& expiration) noexcept
{
    const auto val_days = std::chrono::sys_days{valuation};
    const auto exp_days = std::chrono::sys_days{expiration};
    return static_cast<double>((exp_days - val_days).count()) / 365.0;
}

// -----------------------------------------------------------------------------
// Moneyness computation
// -----------------------------------------------------------------------------

double moneyness_value(
    double strike,
    double spot,
    double rate,
    double div_yield,
    double T,
    Moneyness convention) noexcept
{
    switch (convention) {
        case Moneyness::Simple:
            return strike / spot;
        case Moneyness::LogSimple:
            return std::log(strike / spot);
        case Moneyness::LogForward: {
            const double forward = spot * std::exp((rate - div_yield) * T);
            return std::log(strike / forward);
        }
    }
    return 0.0;  // unreachable — quiets warnings
}

// -----------------------------------------------------------------------------
// Filter: keep the calibrated results, split by expiration in ascending order
// -----------------------------------------------------------------------------

using ExpirationGroup =
    std::pair<std::chrono::year_month_day, std::vector<const CalibrationResult*>>;

std::vector<ExpirationGroup> group_by_expiration(const CalibrationReport& report) {
    // std::map keeps keys sorted for us — expirations natively compare
    // via `year_month_day::operator<=>` (C++20).
    std::map<std::chrono::year_month_day, std::vector<const CalibrationResult*>> buckets;
    for (const auto& r : report.results) {
        if (!r.was_calibrated()) continue;
        buckets[r.option.expiration].push_back(&r);
    }
    std::vector<ExpirationGroup> groups;
    groups.reserve(buckets.size());
    for (auto& [exp, ptrs] : buckets) {
        // Order call-before-put at the same strike, else strike-ascending.
        std::sort(ptrs.begin(), ptrs.end(),
                  [](const CalibrationResult* a, const CalibrationResult* b) {
                      if (a->option.strike != b->option.strike) {
                          return a->option.strike < b->option.strike;
                      }
                      return a->option.type == OptionType::Call
                          && b->option.type == OptionType::Put;
                  });
        groups.emplace_back(exp, std::move(ptrs));
    }
    return groups;
}

// -----------------------------------------------------------------------------
// Linear interpolation on a set of (x, y) points, x strictly ascending.
// Returns nullopt if `target` is outside [x.front(), x.back()] — no
// extrapolation.
// -----------------------------------------------------------------------------

std::optional<double> linear_interpolate(
    std::span<const double> xs,
    std::span<const double> ys,
    double target) noexcept
{
    if (xs.empty() || xs.size() != ys.size()) return std::nullopt;
    if (target < xs.front() || target > xs.back()) return std::nullopt;

    // Locate the first x strictly greater than target.
    const auto it = std::upper_bound(xs.begin(), xs.end(), target);
    if (it == xs.begin()) {
        // target == xs.front() exactly (upper_bound is strict).
        return ys.front();
    }
    if (it == xs.end()) {
        // target == xs.back() exactly.
        return ys.back();
    }
    const std::size_t i_hi = static_cast<std::size_t>(it - xs.begin());
    const std::size_t i_lo = i_hi - 1;

    const double x_lo = xs[i_lo], x_hi = xs[i_hi];
    const double y_lo = ys[i_lo], y_hi = ys[i_hi];
    if (x_hi == x_lo) return y_lo;
    const double t = (target - x_lo) / (x_hi - x_lo);
    return y_lo + t * (y_hi - y_lo);
}

// -----------------------------------------------------------------------------
// numpy-style linear-interpolation percentile
// -----------------------------------------------------------------------------

double linear_percentile(std::vector<double>& sorted_asc, double p) noexcept {
    const auto n = sorted_asc.size();
    if (n == 0) return 0.0;
    if (n == 1) return sorted_asc[0];
    const double rank = (p / 100.0) * static_cast<double>(n - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(rank));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(rank));
    if (lo == hi) return sorted_asc[lo];
    const double frac = rank - static_cast<double>(lo);
    return sorted_asc[lo] + frac * (sorted_asc[hi] - sorted_asc[lo]);
}

// -----------------------------------------------------------------------------
// Statistics core
// -----------------------------------------------------------------------------

IVStatistics compute_stats_from_vector(std::vector<double> values) {
    IVStatistics s{};
    // Drop non-finite values silently — the caller may have handed us
    // NaNs from missing surface cells or failed solves.
    values.erase(
        std::remove_if(values.begin(), values.end(),
                       [](double x) { return !std::isfinite(x); }),
        values.end());

    s.count = values.size();
    if (s.count == 0) return s;

    std::sort(values.begin(), values.end());

    s.min = values.front();
    s.max = values.back();
    s.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(s.count);

    // Median: p50 of the sorted vector. Reuse the percentile helper for
    // consistency.
    s.median = linear_percentile(values, 50.0);
    s.p10    = linear_percentile(values, 10.0);
    s.p25    = linear_percentile(values, 25.0);
    s.p75    = linear_percentile(values, 75.0);
    s.p90    = linear_percentile(values, 90.0);

    if (s.count >= 2) {
        double sum_sq = 0.0;
        for (double v : values) {
            const double d = v - s.mean;
            sum_sq += d * d;
        }
        s.stddev = std::sqrt(sum_sq / static_cast<double>(s.count));
    }
    return s;
}

// -----------------------------------------------------------------------------
// Black-Scholes delta for a single (K, IV) pair using the shipped engine.
// Delegating to the engine keeps the delta formula in one place.
// -----------------------------------------------------------------------------

double bs_delta(
    double S, double K, double r, double q, double T,
    double sigma, OptionType type)
{
    BlackScholesEngine engine;
    return engine.price(BlackScholesEngine::Inputs{
        .spot           = S,
        .strike         = K,
        .rate           = r,
        .dividend_yield = q,
        .volatility     = sigma,
        .time_to_expiry = T,
        .type           = type,
    }).greeks.delta;
}

// -----------------------------------------------------------------------------
// OTM selection: for a given (strike, spot), which type is "OTM"?
// -----------------------------------------------------------------------------

constexpr bool is_otm(double strike, double spot, OptionType type) noexcept {
    return type == OptionType::Call ? (strike >= spot) : (strike <= spot);
}

}  // namespace

// =============================================================================
// build_smiles
// =============================================================================

std::vector<VolatilitySmile> build_smiles(
    const CalibrationReport& report,
    const MarketSnapshot& market,
    Moneyness convention)
{
    const auto groups = group_by_expiration(report);

    std::vector<VolatilitySmile> out;
    out.reserve(groups.size());

    for (const auto& [exp, ptrs] : groups) {
        VolatilitySmile s{};
        s.expiration = exp;
        s.time_to_expiry = year_fraction(market.valuation_date, exp);
        s.moneyness_convention = convention;
        s.strikes.reserve(ptrs.size());
        s.types.reserve(ptrs.size());
        s.moneyness.reserve(ptrs.size());
        s.implied_volatility.reserve(ptrs.size());

        for (const CalibrationResult* r : ptrs) {
            const double K = r->option.strike;
            s.strikes.push_back(K);
            s.types.push_back(r->option.type);
            s.moneyness.push_back(moneyness_value(
                K, market.spot,
                market.risk_free_rate, market.dividend_yield,
                s.time_to_expiry, convention));
            s.implied_volatility.push_back(*r->computed_iv);
        }
        out.push_back(std::move(s));
    }
    return out;
}

// =============================================================================
// build_term_structure
// =============================================================================

TermStructure build_term_structure(
    const CalibrationReport& report,
    const MarketSnapshot& market)
{
    // ATM IV per expiration, interpolated linearly in strike between the
    // two calibrated OTM contracts bracketing spot. If either the lower
    // or upper bracket is missing (e.g. the chain is all-OTM-calls or
    // the spot is outside the range of listed strikes), we emit NaN
    // rather than extrapolating.
    const auto groups = group_by_expiration(report);

    TermStructure term{};
    for (const auto& [exp, ptrs] : groups) {
        term.expirations.push_back(exp);
        term.maturities.push_back(year_fraction(market.valuation_date, exp));

        // Build the (strike -> IV) curve using the OTM convention so the
        // linear interpolation is single-valued at each strike.
        std::map<double, double> otm_curve;
        for (const CalibrationResult* r : ptrs) {
            const double K = r->option.strike;
            const bool otm = is_otm(K, market.spot, r->option.type);
            // Prefer OTM. If we've already seen this strike, only overwrite
            // if the new contract is OTM (won't happen if we picked OTM
            // first, but harmless).
            auto it = otm_curve.find(K);
            if (it == otm_curve.end()) {
                if (otm) {
                    otm_curve.emplace(K, *r->computed_iv);
                } else {
                    // Non-OTM at ATM-and-below/above: still record as a
                    // fallback in case no OTM is available at this strike.
                    otm_curve.emplace(K, *r->computed_iv);
                }
            } else if (otm) {
                it->second = *r->computed_iv;
            }
        }
        // Flatten to parallel vectors ascending in strike.
        std::vector<double> ks; std::vector<double> ivs;
        ks.reserve(otm_curve.size());
        ivs.reserve(otm_curve.size());
        for (const auto& [k, iv] : otm_curve) {
            ks.push_back(k);
            ivs.push_back(iv);
        }
        const auto atm = linear_interpolate(ks, ivs, market.spot);
        term.atm_iv.push_back(atm.value_or(std::numeric_limits<double>::quiet_NaN()));
    }
    return term;
}

// =============================================================================
// build_surface
// =============================================================================

VolatilitySurface build_surface(
    const CalibrationReport& report,
    const MarketSnapshot& market)
{
    const auto groups = group_by_expiration(report);

    // Collect the sorted union of every calibrated strike.
    std::set<double> strike_set;
    for (const auto& [exp, ptrs] : groups) {
        for (const CalibrationResult* r : ptrs) {
            strike_set.insert(r->option.strike);
        }
    }

    VolatilitySurface surface{};
    surface.expirations.reserve(groups.size());
    surface.maturities.reserve(groups.size());
    surface.strikes.assign(strike_set.begin(), strike_set.end());
    surface.implied_vols.reserve(groups.size());

    // Column index by strike for O(log n) lookup during row population.
    std::map<double, std::size_t> strike_to_col;
    for (std::size_t j = 0; j < surface.strikes.size(); ++j) {
        strike_to_col.emplace(surface.strikes[j], j);
    }

    for (const auto& [exp, ptrs] : groups) {
        surface.expirations.push_back(exp);
        surface.maturities.push_back(year_fraction(market.valuation_date, exp));

        std::vector<double> row(surface.strikes.size(),
                                std::numeric_limits<double>::quiet_NaN());

        // Fill: for each calibrated contract at this expiration, prefer
        // OTM; if the same-strike cell is already occupied by a non-OTM
        // fallback, overwrite it with the OTM value.
        std::vector<bool> is_otm_cell(surface.strikes.size(), false);
        for (const CalibrationResult* r : ptrs) {
            const double K = r->option.strike;
            const std::size_t j = strike_to_col.at(K);
            const bool otm = is_otm(K, market.spot, r->option.type);
            if (std::isnan(row[j])) {
                row[j] = *r->computed_iv;
                is_otm_cell[j] = otm;
            } else if (otm && !is_otm_cell[j]) {
                // Replace the non-OTM value with the OTM one.
                row[j] = *r->computed_iv;
                is_otm_cell[j] = true;
            }
            // If we already have an OTM value here and this one isn't
            // OTM, keep what we have.
        }
        surface.implied_vols.push_back(std::move(row));
    }
    return surface;
}

// =============================================================================
// compute_skew_metrics
// =============================================================================

namespace {

/**
 * Interpolate a smile in delta-space to find the IV at a specific target
 * delta. Filters the smile to a single option-type first (calls or puts)
 * because their delta ranges are disjoint. Uses the calibrated IV of
 * each point to compute its BS delta.
 *
 * Returns nullopt when the target delta falls outside the calibrated
 * delta range.
 */
std::optional<double> iv_at_delta(
    const VolatilitySmile& smile,
    const MarketSnapshot& market,
    double target_delta,
    OptionType filter_type)
{
    // Assemble (delta, iv) pairs for the requested type only.
    std::vector<std::pair<double, double>> pts;
    pts.reserve(smile.size());
    for (std::size_t i = 0; i < smile.size(); ++i) {
        if (smile.types[i] != filter_type) continue;
        const double d = bs_delta(
            market.spot, smile.strikes[i],
            market.risk_free_rate, market.dividend_yield,
            smile.time_to_expiry, smile.implied_volatility[i],
            filter_type);
        pts.emplace_back(d, smile.implied_volatility[i]);
    }
    if (pts.size() < 2) return std::nullopt;

    // Sort ascending in delta.
    std::sort(pts.begin(), pts.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Split into parallel spans for the interpolator.
    std::vector<double> ds, vs;
    ds.reserve(pts.size()); vs.reserve(pts.size());
    for (const auto& [d, v] : pts) { ds.push_back(d); vs.push_back(v); }
    return linear_interpolate(ds, vs, target_delta);
}

/**
 * ATM IV: linear-interpolate in strike between the two calibrated OTM
 * strikes bracketing spot, using the same OTM convention as the surface.
 */
std::optional<double> atm_iv_from_smile(
    const VolatilitySmile& smile,
    double spot)
{
    // Build a per-strike OTM curve so a call-and-put at the same strike
    // reduces to a single point.
    std::map<double, double> otm_curve;
    std::map<double, bool>   is_otm_at;
    for (std::size_t i = 0; i < smile.size(); ++i) {
        const double K = smile.strikes[i];
        const bool otm = is_otm(K, spot, smile.types[i]);
        auto it = otm_curve.find(K);
        if (it == otm_curve.end()) {
            otm_curve.emplace(K, smile.implied_volatility[i]);
            is_otm_at[K] = otm;
        } else if (otm && !is_otm_at[K]) {
            it->second = smile.implied_volatility[i];
            is_otm_at[K] = true;
        }
    }
    std::vector<double> ks, vs;
    ks.reserve(otm_curve.size());
    vs.reserve(otm_curve.size());
    for (const auto& [k, v] : otm_curve) { ks.push_back(k); vs.push_back(v); }
    return linear_interpolate(ks, vs, spot);
}

}  // namespace

std::vector<SkewMetrics> compute_skew_metrics(
    std::span<const VolatilitySmile> smiles,
    const MarketSnapshot& market)
{
    std::vector<SkewMetrics> out;
    out.reserve(smiles.size());

    for (const auto& s : smiles) {
        SkewMetrics m{};
        m.expiration     = s.expiration;
        m.time_to_expiry = s.time_to_expiry;

        m.atm_iv          = atm_iv_from_smile(s, market.spot);
        m.call_25delta_iv = iv_at_delta(s, market,  0.25, OptionType::Call);
        m.put_25delta_iv  = iv_at_delta(s, market, -0.25, OptionType::Put);

        if (m.call_25delta_iv && m.put_25delta_iv) {
            m.risk_reversal = *m.call_25delta_iv - *m.put_25delta_iv;
            if (m.atm_iv) {
                m.butterfly =
                    0.5 * (*m.call_25delta_iv + *m.put_25delta_iv) - *m.atm_iv;
            }
        }
        out.push_back(std::move(m));
    }
    return out;
}

// =============================================================================
// Statistics
// =============================================================================

IVStatistics compute_statistics(std::span<const double> values) {
    return compute_stats_from_vector({values.begin(), values.end()});
}

IVStatistics compute_statistics(const CalibrationReport& report) {
    std::vector<double> ivs;
    ivs.reserve(report.results.size());
    for (const auto& r : report.results) {
        if (r.was_calibrated()) ivs.push_back(*r.computed_iv);
    }
    return compute_stats_from_vector(std::move(ivs));
}

std::vector<std::pair<std::chrono::year_month_day, IVStatistics>>
compute_statistics_by_expiration(const CalibrationReport& report)
{
    std::map<std::chrono::year_month_day, std::vector<double>> buckets;
    for (const auto& r : report.results) {
        if (!r.was_calibrated()) continue;
        buckets[r.option.expiration].push_back(*r.computed_iv);
    }
    std::vector<std::pair<std::chrono::year_month_day, IVStatistics>> out;
    out.reserve(buckets.size());
    for (auto& [exp, vs] : buckets) {
        out.emplace_back(exp, compute_stats_from_vector(std::move(vs)));
    }
    return out;
}

TypeStatistics compute_statistics_by_type(const CalibrationReport& report) {
    std::vector<double> calls, puts;
    for (const auto& r : report.results) {
        if (!r.was_calibrated()) continue;
        if (r.option.type == OptionType::Call) {
            calls.push_back(*r.computed_iv);
        } else {
            puts.push_back(*r.computed_iv);
        }
    }
    return TypeStatistics{
        .calls = compute_stats_from_vector(std::move(calls)),
        .puts  = compute_stats_from_vector(std::move(puts)),
    };
}

// =============================================================================
// CSV export — smiles
// =============================================================================

namespace {

constexpr std::string_view kSmileHeader =
    "expiration,time_to_expiry,strike,option_type,"
    "moneyness_convention,moneyness,implied_volatility\n";

void write_smile_row(const VolatilitySmile& smile, std::size_t i, std::ostream& os) {
    os << format_date(smile.expiration) << ','
       << format_double(smile.time_to_expiry) << ','
       << format_double(smile.strikes[i]) << ','
       << type_string(smile.types[i]) << ','
       << to_string(smile.moneyness_convention) << ','
       << format_double(smile.moneyness[i]) << ','
       << format_double(smile.implied_volatility[i]) << '\n';
}

}  // namespace

void write_csv(const VolatilitySmile& smile, std::ostream& os) {
    os << kSmileHeader;
    for (std::size_t i = 0; i < smile.size(); ++i) {
        write_smile_row(smile, i, os);
    }
}

void write_csv(std::span<const VolatilitySmile> smiles, std::ostream& os) {
    os << kSmileHeader;
    for (const auto& s : smiles) {
        for (std::size_t i = 0; i < s.size(); ++i) {
            write_smile_row(s, i, os);
        }
    }
}

void write_csv(std::span<const VolatilitySmile> smiles,
               const std::filesystem::path& path)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "write_csv(smiles): failed to open " + path.string());
    }
    write_csv(smiles, out);
}

// =============================================================================
// CSV export — term structure
// =============================================================================

void write_csv(const TermStructure& term, std::ostream& os) {
    os << "expiration,time_to_expiry,atm_iv\n";
    for (std::size_t i = 0; i < term.size(); ++i) {
        os << format_date(term.expirations[i]) << ','
           << format_double(term.maturities[i]) << ','
           << format_double(term.atm_iv[i]) << '\n';
    }
}

void write_csv(const TermStructure& term, const std::filesystem::path& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "write_csv(term): failed to open " + path.string());
    }
    write_csv(term, out);
}

// =============================================================================
// CSV export — surface (long format)
// =============================================================================

void write_csv(const VolatilitySurface& surface, std::ostream& os) {
    os << "expiration,time_to_expiry,strike,implied_volatility\n";
    for (std::size_t i = 0; i < surface.rows(); ++i) {
        for (std::size_t j = 0; j < surface.cols(); ++j) {
            os << format_date(surface.expirations[i]) << ','
               << format_double(surface.maturities[i]) << ','
               << format_double(surface.strikes[j]) << ','
               << format_double(surface.implied_vols[i][j]) << '\n';
        }
    }
}

void write_csv(const VolatilitySurface& surface, const std::filesystem::path& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "write_csv(surface): failed to open " + path.string());
    }
    write_csv(surface, out);
}

// =============================================================================
// CSV export — skew metrics
// =============================================================================

void write_csv(std::span<const SkewMetrics> skews, std::ostream& os) {
    os << "expiration,time_to_expiry,atm_iv,call_25delta_iv,put_25delta_iv,"
          "risk_reversal,butterfly\n";
    for (const auto& s : skews) {
        os << format_date(s.expiration) << ','
           << format_double(s.time_to_expiry) << ','
           << format_optional(s.atm_iv) << ','
           << format_optional(s.call_25delta_iv) << ','
           << format_optional(s.put_25delta_iv) << ','
           << format_optional(s.risk_reversal) << ','
           << format_optional(s.butterfly) << '\n';
    }
}

void write_csv(std::span<const SkewMetrics> skews, const std::filesystem::path& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "write_csv(skews): failed to open " + path.string());
    }
    write_csv(skews, out);
}

}  // namespace ore::analytics
