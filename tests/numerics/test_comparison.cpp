#include <gtest/gtest.h>

#include <ore/numerics/comparison.hpp>

#include <cmath>
#include <limits>

using ore::numerics::absolute_error;
using ore::numerics::approximately_equal;
using ore::numerics::relative_error;

// ---- absolute_error -------------------------------------------------------

TEST(AbsoluteErrorTest, IsSymmetric) {
    EXPECT_DOUBLE_EQ(absolute_error(3.0, 5.0), 2.0);
    EXPECT_DOUBLE_EQ(absolute_error(5.0, 3.0), 2.0);
}

TEST(AbsoluteErrorTest, IsZeroForEqualValues) {
    EXPECT_DOUBLE_EQ(absolute_error(0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(absolute_error(-1.5, -1.5), 0.0);
    EXPECT_DOUBLE_EQ(absolute_error(1e100, 1e100), 0.0);
}

TEST(AbsoluteErrorTest, HandlesSignedInputs) {
    EXPECT_DOUBLE_EQ(absolute_error(-2.0, 3.0), 5.0);
    EXPECT_DOUBLE_EQ(absolute_error(-5.0, -1.0), 4.0);
}

TEST(AbsoluteErrorTest, PropagatesNaN) {
    EXPECT_TRUE(std::isnan(absolute_error(
        std::numeric_limits<double>::quiet_NaN(), 1.0)));
}

// ---- relative_error -------------------------------------------------------

TEST(RelativeErrorTest, IsSymmetric) {
    EXPECT_DOUBLE_EQ(relative_error(100.0, 101.0), 1.0 / 101.0);
    EXPECT_DOUBLE_EQ(relative_error(101.0, 100.0), 1.0 / 101.0);
}

TEST(RelativeErrorTest, ScalesWithMagnitude) {
    // Scaling invariance: relative_error(k*a, k*b) == relative_error(a, b).
    // For (100, 101), the metric is |1| / max(100, 101) = 1/101, and the
    // same value should appear for (1, 1.01) and (1e6, 1.01e6) up to
    // rounding.
    const double baseline = 1.0 / 101.0;
    EXPECT_NEAR(relative_error(100.0, 101.0), baseline, 1e-15);
    EXPECT_NEAR(relative_error(1.0, 1.01),    baseline, 1e-15);
    EXPECT_NEAR(relative_error(1e6, 1.01e6),  baseline, 1e-15);
}

TEST(RelativeErrorTest, IsZeroWhenBothOperandsAreZero) {
    // We define 0/0 as 0 here — see comparison.hpp for rationale.
    EXPECT_DOUBLE_EQ(relative_error(0.0, 0.0), 0.0);
}

TEST(RelativeErrorTest, IsOneWhenOnlyOneOperandIsZero) {
    // |a - 0| / max(|a|, 0) = |a| / |a| = 1.
    EXPECT_DOUBLE_EQ(relative_error(5.0, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(relative_error(0.0, -5.0), 1.0);
}

TEST(RelativeErrorTest, PropagatesNaN) {
    EXPECT_TRUE(std::isnan(relative_error(
        std::numeric_limits<double>::quiet_NaN(), 1.0)));
}

// ---- approximately_equal --------------------------------------------------

TEST(ApproximatelyEqualTest, ExactMatchIsAlwaysEqual) {
    EXPECT_TRUE(approximately_equal(1.0, 1.0, 0.0, 0.0));
    EXPECT_TRUE(approximately_equal(0.0, 0.0, 0.0, 0.0));
    EXPECT_TRUE(approximately_equal(-1e100, -1e100, 0.0, 0.0));
}

TEST(ApproximatelyEqualTest, AbsoluteToleranceHandlesNearZeroValues) {
    // Values that would fail a pure relative comparison (denominator ~ 0)
    // pass with a modest absolute tolerance.
    EXPECT_TRUE(approximately_equal(1e-15, -1e-15, 1e-12, 0.0));
    EXPECT_FALSE(approximately_equal(1e-10, -1e-10, 1e-12, 0.0));
}

TEST(ApproximatelyEqualTest, RelativeToleranceHandlesLargeValues) {
    // Values that would fail a pure absolute comparison (huge magnitude
    // means small relative differences look large in absolute terms)
    // pass with a relative tolerance.
    EXPECT_TRUE(approximately_equal(1e12, 1.000001e12, 0.0, 1e-5));
    EXPECT_FALSE(approximately_equal(1e12, 1.001e12, 0.0, 1e-5));
}

TEST(ApproximatelyEqualTest, CombinedToleranceIsUnionOfBoth) {
    // A pair that fails abs_tol alone AND fails rel_tol alone but passes
    // once you combine the two (their sum exceeds |a - b|).
    // |a-b| = 0.0011, abs_tol=0.001, rel_tol=1e-4, max(|a|,|b|) ~= 1.001
    // combined bound = 0.001 + 1e-4 * 1.001 = 0.001 + 0.0001001 = 0.0011001
    // 0.0011 <= 0.0011001 -> true
    EXPECT_TRUE(approximately_equal(1.0, 1.0011, 0.001, 1e-4));
}

TEST(ApproximatelyEqualTest, NaNIsNeverEqualToAnything) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(approximately_equal(nan, nan, 1e10, 1e10));
    EXPECT_FALSE(approximately_equal(nan, 0.0, 1e10, 1e10));
    EXPECT_FALSE(approximately_equal(0.0, nan, 1e10, 1e10));
}

TEST(ApproximatelyEqualTest, InfinityRules) {
    const double inf = std::numeric_limits<double>::infinity();
    // +inf compared to +inf: |inf - inf| = NaN, so combined bound check is
    // NaN <= anything which is false. That's fine and documented.
    EXPECT_FALSE(approximately_equal(inf, inf, 1.0, 1.0));
    EXPECT_FALSE(approximately_equal(inf, 1.0, 1.0, 1.0));
    EXPECT_FALSE(approximately_equal(1.0, -inf, 1.0, 1.0));
}
