#include <gtest/gtest.h>

#include <ore/numerics/constants.hpp>

#include <cmath>
#include <numbers>

namespace nc = ore::numerics::constants;

TEST(NumericsConstantsTest, TwoPiMatchesStdNumbers) {
    EXPECT_DOUBLE_EQ(nc::two_pi, 2.0 * std::numbers::pi_v<double>);
}

TEST(NumericsConstantsTest, SqrtTwoPiMatchesStdlibComputation) {
    // Comparing to a std::sqrt of a compile-time expression is a good
    // ULP-level cross-check for the literal digits.
    EXPECT_NEAR(nc::sqrt_two_pi,
                std::sqrt(2.0 * std::numbers::pi_v<double>),
                1e-15);
}

TEST(NumericsConstantsTest, InvSqrtTwoPiIsTheReciprocal) {
    EXPECT_NEAR(nc::inv_sqrt_two_pi * nc::sqrt_two_pi, 1.0, 1e-15);
}

TEST(NumericsConstantsTest, ConstantsAreConstexpr) {
    // If any of these were not constexpr the following declarations would
    // fail to compile.
    constexpr double a = nc::two_pi;
    constexpr double b = nc::sqrt_two_pi;
    constexpr double c = nc::inv_sqrt_two_pi;
    EXPECT_GT(a, 0.0);
    EXPECT_GT(b, 0.0);
    EXPECT_GT(c, 0.0);
}
