#include <gtest/gtest.h>

#include <ore/numerics/normal_distribution.hpp>

#include <cmath>
#include <limits>

using ore::numerics::NormalDistribution;

// ---- PDF ------------------------------------------------------------------

TEST(NormalPdfTest, PdfAtZeroIsInvSqrt2Pi) {
    // phi(0) = 1 / sqrt(2*pi) = 0.39894228040143267...
    EXPECT_NEAR(NormalDistribution::pdf(0.0), 0.398942280401432677939946059934,
                1e-15);
}

TEST(NormalPdfTest, PdfIsSymmetric) {
    for (double x : {0.1, 0.5, 1.0, 1.96, 3.0, 5.0}) {
        EXPECT_DOUBLE_EQ(NormalDistribution::pdf(x),
                         NormalDistribution::pdf(-x))
            << "asymmetry at x=" << x;
    }
}

TEST(NormalPdfTest, PdfIsMonotonicallyDecreasingForPositiveX) {
    double last = std::numeric_limits<double>::infinity();
    for (double x = 0.0; x < 10.0; x += 0.25) {
        const double v = NormalDistribution::pdf(x);
        EXPECT_LE(v, last) << "pdf increased at x=" << x;
        last = v;
    }
}

TEST(NormalPdfTest, PdfMatchesExplicitFormula) {
    // phi(x) = (1/sqrt(2*pi)) * exp(-x*x/2)
    constexpr double inv_sqrt_2pi = 0.3989422804014327;
    for (double x : {-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0}) {
        const double expected = inv_sqrt_2pi * std::exp(-0.5 * x * x);
        EXPECT_NEAR(NormalDistribution::pdf(x), expected, 1e-14);
    }
}

TEST(NormalPdfTest, PdfUnderflowsToZeroForExtremeInput) {
    // exp(-x*x/2) for x = 40 is ~10^-348, well past double's underflow.
    EXPECT_EQ(NormalDistribution::pdf(40.0), 0.0);
    EXPECT_EQ(NormalDistribution::pdf(-40.0), 0.0);
}

TEST(NormalPdfTest, PdfPropagatesNaN) {
    EXPECT_TRUE(std::isnan(NormalDistribution::pdf(
        std::numeric_limits<double>::quiet_NaN())));
}

// ---- CDF ------------------------------------------------------------------

TEST(NormalCdfTest, CdfAtZeroIsOneHalf) {
    EXPECT_DOUBLE_EQ(NormalDistribution::cdf(0.0), 0.5);
}

TEST(NormalCdfTest, CdfSymmetryAroundOneHalf) {
    // Phi(-x) + Phi(x) = 1
    for (double x : {0.1, 0.5, 1.0, 1.96, 3.0, 5.0, 8.0}) {
        EXPECT_NEAR(NormalDistribution::cdf(-x) + NormalDistribution::cdf(x),
                    1.0, 1e-14)
            << "symmetry broken at x=" << x;
    }
}

TEST(NormalCdfTest, CdfKnownValues) {
    // Classic textbook values.
    struct Point { double x; double phi; };
    const Point points[] = {
        {-3.0, 0.001349898031630094},
        {-1.96, 0.024997895148220487},
        {-1.0, 0.158655253931457108},
        {-0.5, 0.308537538725986990},
        { 0.0, 0.500000000000000000},
        { 0.5, 0.691462461274013010},
        { 1.0, 0.841344746068542892},
        { 1.96, 0.975002104851779513},
        { 3.0, 0.998650101968369906},
    };
    for (const auto& p : points) {
        EXPECT_NEAR(NormalDistribution::cdf(p.x), p.phi, 1e-13)
            << "at x=" << p.x;
    }
}

TEST(NormalCdfTest, CdfIsMonotonicallyIncreasing) {
    double last = 0.0;
    for (double x = -6.0; x <= 6.0; x += 0.1) {
        const double v = NormalDistribution::cdf(x);
        EXPECT_GE(v, last) << "cdf decreased at x=" << x;
        last = v;
    }
}

TEST(NormalCdfTest, CdfSaturatesInTails) {
    // Deep in the tails, Phi(x) is representationally 0 or 1.
    EXPECT_EQ(NormalDistribution::cdf(-40.0), 0.0);
    EXPECT_EQ(NormalDistribution::cdf(40.0), 1.0);
    // Milder tails should still be strictly inside (0, 1).
    EXPECT_GT(NormalDistribution::cdf(-8.0), 0.0);
    EXPECT_LT(NormalDistribution::cdf(8.0), 1.0);
}

TEST(NormalCdfTest, CdfPreservesTailPrecision) {
    // Direct evaluation of Phi(-6) ~ 9.87e-10 — representable as a
    // full-precision double. Our erfc-based formulation computes this
    // directly; a hypothetical `0.5 * (1 + erf(x/sqrt(2)))` implementation
    // would round Phi(6) to 0.9999999990... and lose every significant
    // digit of the tail if the caller ever needed `1 - Phi(6)`.
    // Reference: scipy.stats.norm.cdf(-6) = 9.865876450376983e-10.
    const double phi_minus_6 = NormalDistribution::cdf(-6.0);
    EXPECT_NEAR(phi_minus_6, 9.865876450376983e-10, 1e-20);
}

TEST(NormalCdfTest, CdfPropagatesNaN) {
    EXPECT_TRUE(std::isnan(NormalDistribution::cdf(
        std::numeric_limits<double>::quiet_NaN())));
}

// ---- Inverse CDF (Wichura AS241) ------------------------------------------

TEST(NormalInverseCdfTest, MedianIsZero) {
    EXPECT_DOUBLE_EQ(NormalDistribution::inverse_cdf(0.5), 0.0);
}

TEST(NormalInverseCdfTest, KnownQuantiles) {
    // Values verified against R's qnorm() and scipy.stats.norm.ppf().
    struct Point { double p; double x; };
    const Point points[] = {
        {0.001, -3.090232306167813 },
        {0.010, -2.326347874040841 },
        {0.025, -1.959963984540054 },
        {0.050, -1.644853626951472 },
        {0.100, -1.281551565544600 },
        {0.250, -0.674489750196082 },
        {0.500,  0.0                },
        {0.750,  0.674489750196082 },
        {0.900,  1.281551565544600 },
        {0.950,  1.644853626951472 },
        {0.975,  1.959963984540054 },
        {0.990,  2.326347874040841 },
        {0.999,  3.090232306167813 },
    };
    for (const auto& p : points) {
        EXPECT_NEAR(NormalDistribution::inverse_cdf(p.p), p.x, 1e-12)
            << "at p=" << p.p;
    }
}

TEST(NormalInverseCdfTest, SymmetryAroundOneHalf) {
    // Phi^-1(1 - p) = -Phi^-1(p)
    for (double p : {1e-6, 1e-3, 0.01, 0.1, 0.25, 0.4, 0.499}) {
        const double left  = NormalDistribution::inverse_cdf(p);
        const double right = NormalDistribution::inverse_cdf(1.0 - p);
        EXPECT_NEAR(right, -left, 1e-13) << "asymmetry at p=" << p;
    }
}

TEST(NormalInverseCdfTest, RoundTripsThroughCdf) {
    // cdf(inverse_cdf(p)) ~= p across the whole (0, 1) interval.
    // We accept a small absolute error because the composition goes
    // through erfc, exp, log, sqrt — each contributes ~1 ULP of noise.
    for (double p : {1e-9, 1e-6, 1e-3, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9,
                     0.99, 0.999, 0.999999, 1.0 - 1e-9}) {
        const double x = NormalDistribution::inverse_cdf(p);
        const double round_trip = NormalDistribution::cdf(x);
        EXPECT_NEAR(round_trip, p, 1e-13) << "round-trip fails at p=" << p;
    }
}

TEST(NormalInverseCdfTest, IsMonotonicallyIncreasing) {
    double last = -std::numeric_limits<double>::infinity();
    for (double p = 0.001; p < 1.0; p += 0.005) {
        const double x = NormalDistribution::inverse_cdf(p);
        EXPECT_GT(x, last) << "non-monotone at p=" << p;
        last = x;
    }
}

TEST(NormalInverseCdfTest, TailBoundaryContinuity) {
    // Central-vs-tail boundary is at |q| = 0.425, i.e. p in
    // {0.075, 0.925}. Values immediately on either side should be within
    // a few ULP of each other — the two rational approximations were
    // designed to be numerically continuous at the boundary.
    const double eps = 1e-6;
    const double a = NormalDistribution::inverse_cdf(0.075 - eps);
    const double b = NormalDistribution::inverse_cdf(0.075 + eps);
    EXPECT_NEAR(a, b, 1e-4);  // very close over a 2*eps p-interval

    const double c = NormalDistribution::inverse_cdf(0.925 - eps);
    const double d = NormalDistribution::inverse_cdf(0.925 + eps);
    EXPECT_NEAR(c, d, 1e-4);
}

TEST(NormalInverseCdfTest, ExtremeTailBoundaryStaysAccurate) {
    // The moderate/extreme-tail boundary is at r = sqrt(-log(p)) = 5,
    // i.e. p ~ exp(-25) ~ 1.4e-11. Both branches should give consistent
    // values against known R quantiles just inside/outside that region.
    // R: qnorm(1e-11) ~ -6.706, qnorm(1e-13) ~ -7.349
    EXPECT_NEAR(NormalDistribution::inverse_cdf(1e-11), -6.7060231554951654, 1e-9);
    EXPECT_NEAR(NormalDistribution::inverse_cdf(1e-13), -7.3487545100050696, 1e-9);
}

TEST(NormalInverseCdfTest, EndpointsReturnInfinity) {
    EXPECT_EQ(NormalDistribution::inverse_cdf(0.0),
              -std::numeric_limits<double>::infinity());
    EXPECT_EQ(NormalDistribution::inverse_cdf(1.0),
               std::numeric_limits<double>::infinity());
}

TEST(NormalInverseCdfTest, OutOfDomainReturnsNaN) {
    EXPECT_TRUE(std::isnan(NormalDistribution::inverse_cdf(-0.5)));
    EXPECT_TRUE(std::isnan(NormalDistribution::inverse_cdf(1.5)));
    EXPECT_TRUE(std::isnan(NormalDistribution::inverse_cdf(
        std::numeric_limits<double>::quiet_NaN())));
    EXPECT_TRUE(std::isnan(NormalDistribution::inverse_cdf(
        std::numeric_limits<double>::infinity())));
}
