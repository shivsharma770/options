#include <ore/numerics/normal_distribution.hpp>

#include <cmath>
#include <limits>
#include <numbers>

#include <ore/numerics/constants.hpp>

namespace ore::numerics {

namespace {

// -----------------------------------------------------------------------------
// Wichura AS241 (1988) coefficients for the inverse standard-normal CDF.
//
// Reference: Wichura, M. J. (1988), "Algorithm AS 241: The percentage points of
// the normal distribution", Applied Statistics 37(3), pp. 477-484.
// Also implemented as `qnorm5` in R's `nmath/qnorm.c`.
//
// Three regions:
//   Central       : |q| <= 0.425 where q = p - 0.5.
//                   Approximation is rational in r = 0.180625 - q*q.
//   Moderate tail : 0.425 < |q|, and r' = sqrt(-log(min(p, 1-p))) <= 5.
//                   Approximation is rational in r'' = r' - 1.6.
//   Extreme tail  : r' > 5. Approximation is rational in r'' = r' - 5.
//
// The coefficients below are the ones published in the AS241 paper. Every
// polynomial is evaluated with Horner's method for numerical stability.
// -----------------------------------------------------------------------------

// Each polynomial is degree 7 (8 coefficients), highest degree first, and
// the denominator arrays include the trailing "+ 1.0" constant term
// explicitly so numerator and denominator are evaluated the same way.

// Central region (|q| <= 0.425), rational in r = 0.180625 - q*q.
constexpr double kCentralNum[8] = {
    2.509080928730122e+3,
    3.343057558358813e+4,
    6.726577092700870e+4,
    4.592195393154987e+4,
    1.373169376550946e+4,
    1.971590950306551e+3,
    1.331416678917844e+2,
    3.387132872796366e+0,
};
constexpr double kCentralDen[8] = {
    5.226495278852855e+3,
    2.872908573572194e+4,
    3.930789580009271e+4,
    2.121379430158660e+4,
    5.394196021424751e+3,
    6.871870074920579e+2,
    4.231333070160091e+1,
    1.0,
};

// Moderate tail region (r_tail_raw <= 5), rational in r = r_tail_raw - 1.6.
constexpr double kMidNum[8] = {
    7.745450142783414e-4,
    2.272384498926918e-2,
    2.417807251774506e-1,
    1.270458252452368e+0,
    3.647848324763204e+0,
    5.769497221460691e+0,
    4.630337846156545e+0,
    1.423437110749683e+0,
};
constexpr double kMidDen[8] = {
    1.050750071644417e-9,
    5.475938084995345e-4,
    1.519866656361646e-2,
    1.481039764274800e-1,
    6.897673349851000e-1,
    1.676384830183804e+0,
    2.053191626637759e+0,
    1.0,
};

// Extreme tail region (r_tail_raw > 5), rational in r = r_tail_raw - 5.
constexpr double kTailNum[8] = {
    2.010334399292288e-7,
    2.711555568743488e-5,
    1.242660947388078e-3,
    2.653218952657612e-2,
    2.965605718285049e-1,
    1.784826539917291e+0,
    5.463784911164114e+0,
    6.657904643501104e+0,
};
constexpr double kTailDen[8] = {
    2.044263103389940e-15,
    1.421511758316446e-7,
    1.846318317510055e-5,
    7.868691311456133e-4,
    1.487536129085061e-2,
    1.369298809227358e-1,
    5.998322065558879e-1,
    1.0,
};

// Horner-scheme polynomial evaluation. `coeffs` is highest-degree first;
// with N == 8 this is a degree-7 polynomial. The explicit form the AS241
// references print out inline compiles to the same instructions but is
// far harder to audit against the paper.
inline double horner8(const double (&coeffs)[8], double x) noexcept {
    double acc = coeffs[0];
    for (int i = 1; i < 8; ++i) {
        acc = acc * x + coeffs[i];
    }
    return acc;
}

} // unnamed namespace

double NormalDistribution::pdf(double x) noexcept {
    // phi(x) = (1 / sqrt(2*pi)) * exp(-x^2 / 2)
    //
    // The constant is precomputed. `x * x` is preferred over `std::pow(x, 2)`
    // both for accuracy (a single multiply, no transcendental rounding) and
    // speed. For very large |x| the exponent underflows to 0.0, which is
    // the correct IEEE-754 answer for a density that decays faster than
    // any polynomial.
    return constants::inv_sqrt_two_pi * std::exp(-0.5 * x * x);
}

double NormalDistribution::cdf(double x) noexcept {
    // Phi(x) = 0.5 * erfc(-x / sqrt(2))
    //
    // Why `erfc(-x/sqrt(2))` rather than `0.5 * (1 + erf(x/sqrt(2)))`:
    //
    //   * For large positive x, `erf(x/sqrt(2))` is extremely close to 1.
    //     Computing `1 + erf(...)` in `double` loses precision in the
    //     upper tail. Downstream code that later needs `1 - Phi(x)` would
    //     then be looking at zero significant digits.
    //   * `std::erfc` is IEEE-754 spec'd to compute the complementary error
    //     function directly, without going through the cancellation.
    //   * The identity `erf(-y) = -erf(y)` combined with
    //     `erfc(y) = 1 - erf(y)` gives `Phi(x) = 0.5 * erfc(-x/sqrt(2))`
    //     with symmetric accuracy in both tails.
    //
    // The 1/sqrt(2) factor is `std::numbers::sqrt2_v<double> / 2`; using
    // the divide form matches the way most references write the formula.
    constexpr double inv_sqrt_two = 1.0 / std::numbers::sqrt2_v<double>;
    return 0.5 * std::erfc(-x * inv_sqrt_two);
}

double NormalDistribution::inverse_cdf(double p) noexcept {
    // -------------------------------------------------------------------------
    // Domain handling — do it up front, once. Everything below assumes
    // p is a finite value strictly inside (0, 1).
    // -------------------------------------------------------------------------
    if (std::isnan(p)) return std::numeric_limits<double>::quiet_NaN();
    if (p <= 0.0) {
        return p == 0.0 ? -std::numeric_limits<double>::infinity()
                        :  std::numeric_limits<double>::quiet_NaN();
    }
    if (p >= 1.0) {
        return p == 1.0 ?  std::numeric_limits<double>::infinity()
                        :  std::numeric_limits<double>::quiet_NaN();
    }

    const double q = p - 0.5;

    // -------------------------------------------------------------------------
    // Central region: |q| <= 0.425, i.e. p in [0.075, 0.925].
    // Argument to the rational approximation is r = 0.180625 - q*q.
    // Result is q * P(r) / Q(r).
    // -------------------------------------------------------------------------
    if (std::abs(q) <= 0.425) {
        const double r     = 0.180625 - q * q;
        const double numer = horner8(kCentralNum, r);
        const double denom = horner8(kCentralDen, r);
        return q * numer / denom;
    }

    // -------------------------------------------------------------------------
    // Tail regions. Use r' = sqrt(-log(min(p, 1-p))) as the "distance from
    // the tail" — it's monotone in the tail and stays representable down
    // to the smallest positive double.
    //
    // The moderate-tail region uses a rational in r'' = r' - 1.6; the
    // extreme-tail region uses a rational in r'' = r' - 5.0. The shift
    // keeps the polynomial argument near zero on each branch, which is
    // where a low-degree rational function has room to be accurate.
    // -------------------------------------------------------------------------
    const double r_tail_raw = std::sqrt(-std::log(q < 0.0 ? p : 1.0 - p));

    double value;
    if (r_tail_raw <= 5.0) {
        const double r     = r_tail_raw - 1.6;
        const double numer = horner8(kMidNum, r);
        const double denom = horner8(kMidDen, r);
        value = numer / denom;
    } else {
        const double r     = r_tail_raw - 5.0;
        const double numer = horner8(kTailNum, r);
        const double denom = horner8(kTailDen, r);
        value = numer / denom;
    }

    return q < 0.0 ? -value : value;
}

} // namespace ore::numerics
