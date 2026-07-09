#include <gtest/gtest.h>

#include <ore/numerics/bisection.hpp>
#include <ore/numerics/solver_result.hpp>

#include <cmath>
#include <limits>

using ore::numerics::BisectionSolver;
using ore::numerics::SolverResult;
using ore::numerics::SolverStatus;

// ---- Textbook problems ----------------------------------------------------

TEST(BisectionTest, FindsSquareRootOfTwo) {
    const BisectionSolver solver;
    const auto f = [](double x) { return x * x - 2.0; };
    const auto r = solver.solve(f, 1.0, 2.0);
    ASSERT_TRUE(r.converged()) << to_string(r.status);
    EXPECT_NEAR(r.root, std::sqrt(2.0), 1e-9);
}

TEST(BisectionTest, FindsCubicRoot) {
    // f(x) = x^3 - x - 2, root at ~1.5213797068045676
    const BisectionSolver solver;
    const auto f = [](double x) { return x * x * x - x - 2.0; };
    const auto r = solver.solve(f, 1.0, 2.0);
    ASSERT_TRUE(r.converged());
    EXPECT_NEAR(r.root, 1.5213797068045676, 1e-9);
}

TEST(BisectionTest, FindsFixedPointOfCosine) {
    const BisectionSolver solver;
    const auto f = [](double x) { return std::cos(x) - x; };
    const auto r = solver.solve(f, 0.0, 1.0);
    ASSERT_TRUE(r.converged());
    EXPECT_NEAR(r.root, 0.7390851332151607, 1e-9);
}

TEST(BisectionTest, WorksWhenBracketIsGivenReversed) {
    // Callers should be free to pass a > b; the solver swaps internally.
    const BisectionSolver solver;
    const auto f = [](double x) { return x * x - 2.0; };
    const auto r = solver.solve(f, 2.0, 1.0);
    ASSERT_TRUE(r.converged());
    EXPECT_NEAR(r.root, std::sqrt(2.0), 1e-9);
}

TEST(BisectionTest, ShortCircuitsWhenEndpointIsAlreadyARoot) {
    const BisectionSolver solver;
    const auto f = [](double x) { return x - 3.0; };
    const auto r_low  = solver.solve(f, 3.0, 5.0);
    const auto r_high = solver.solve(f, 1.0, 3.0);
    ASSERT_TRUE(r_low.converged());
    ASSERT_TRUE(r_high.converged());
    EXPECT_DOUBLE_EQ(r_low.root, 3.0);
    EXPECT_DOUBLE_EQ(r_high.root, 3.0);
    EXPECT_EQ(r_low.iterations, 0U);
    EXPECT_EQ(r_high.iterations, 0U);
}

// ---- Precision & performance ---------------------------------------------

TEST(BisectionTest, TightToleranceRequiresMoreIterations) {
    BisectionSolver loose({.tolerance = 1e-6});
    BisectionSolver tight({.tolerance = 1e-14});
    const auto f = [](double x) { return x * x - 2.0; };
    const auto r_loose = loose.solve(f, 1.0, 2.0);
    const auto r_tight = tight.solve(f, 1.0, 2.0);
    ASSERT_TRUE(r_loose.converged());
    ASSERT_TRUE(r_tight.converged());
    EXPECT_LT(r_loose.iterations, r_tight.iterations);
    EXPECT_LT(r_tight.residual, 1e-14);
}

TEST(BisectionTest, TerminatesAtFloatingPointPrecisionEvenIfResidualNeverShrinks) {
    // f(x) = 1 for x >= 0, -1 for x < 0. There is no continuous root, but
    // the bracket-collapse guard should return Converged when the
    // midpoint stops moving.
    const BisectionSolver solver{{.tolerance = 1e-300}};
    const auto f = [](double x) { return x < 0.0 ? -1.0 : 1.0; };
    const auto r = solver.solve(f, -1.0, 1.0);
    EXPECT_EQ(r.status, SolverStatus::Converged);
    EXPECT_LT(r.iterations, 100U);
}

// ---- Failure modes --------------------------------------------------------

TEST(BisectionTest, ReportsInvalidBracket) {
    BisectionSolver solver;
    const auto f = [](double x) { return x * x + 1.0; };  // strictly positive
    const auto r = solver.solve(f, -2.0, 2.0);
    EXPECT_EQ(r.status, SolverStatus::InvalidBracket);
    EXPECT_FALSE(r.converged());
    EXPECT_EQ(r.iterations, 0U);
}

TEST(BisectionTest, ReportsMaxIterationsReached) {
    // Configure an unreachable tolerance and a low iteration cap.
    BisectionSolver solver({.tolerance = 1e-300, .max_iterations = 5});
    const auto f = [](double x) { return x * x - 2.0; };
    const auto r = solver.solve(f, 1.0, 2.0);
    EXPECT_EQ(r.status, SolverStatus::MaxIterationsReached);
    EXPECT_EQ(r.iterations, 5U);
    EXPECT_NEAR(r.root, std::sqrt(2.0), 1.0 / 32.0);  // 5 halvings on width 1
}

TEST(BisectionTest, ReportsNonFiniteEvaluationAtEndpoint) {
    BisectionSolver solver;
    const auto f = [](double x) {
        return x == 0.0 ? std::numeric_limits<double>::quiet_NaN() : x - 1.0;
    };
    const auto r = solver.solve(f, 0.0, 2.0);
    EXPECT_EQ(r.status, SolverStatus::NonFiniteEvaluation);
}

TEST(BisectionTest, ReportsNonFiniteEvaluationMidIteration) {
    BisectionSolver solver;
    // f: -1 at x=-1, +1 at x=+1, NaN at x=0 (midpoint after one bisection).
    const auto f = [](double x) {
        if (x == 0.0) return std::numeric_limits<double>::quiet_NaN();
        return x;
    };
    const auto r = solver.solve(f, -1.0, 1.0);
    EXPECT_EQ(r.status, SolverStatus::NonFiniteEvaluation);
}

// ---- Preconditions --------------------------------------------------------

TEST(BisectionTest, ThrowsOnNonFiniteEndpoint) {
    BisectionSolver solver;
    const auto f = [](double x) { return x; };
    EXPECT_THROW(solver.solve(f, std::numeric_limits<double>::quiet_NaN(), 1.0),
                 std::invalid_argument);
    EXPECT_THROW(solver.solve(f, 0.0, std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
}

TEST(BisectionTest, ThrowsOnZeroWidthBracket) {
    BisectionSolver solver;
    const auto f = [](double x) { return x; };
    EXPECT_THROW(solver.solve(f, 1.0, 1.0), std::invalid_argument);
}

TEST(BisectionTest, ThrowsOnNonPositiveTolerance) {
    BisectionSolver solver({.tolerance = 0.0});
    const auto f = [](double x) { return x; };
    EXPECT_THROW(solver.solve(f, -1.0, 1.0), std::invalid_argument);
}

TEST(BisectionTest, ThrowsOnZeroMaxIterations) {
    BisectionSolver solver({.tolerance = 1e-10, .max_iterations = 0});
    const auto f = [](double x) { return x; };
    EXPECT_THROW(solver.solve(f, -1.0, 1.0), std::invalid_argument);
}

// ---- Status stringification ----------------------------------------------

TEST(SolverStatusTest, ToStringCoversEveryEnumerator) {
    using ore::numerics::to_string;
    EXPECT_EQ(to_string(SolverStatus::Converged), "Converged");
    EXPECT_EQ(to_string(SolverStatus::MaxIterationsReached), "MaxIterationsReached");
    EXPECT_EQ(to_string(SolverStatus::InvalidBracket), "InvalidBracket");
    EXPECT_EQ(to_string(SolverStatus::DerivativeTooSmall), "DerivativeTooSmall");
    EXPECT_EQ(to_string(SolverStatus::NonFiniteEvaluation), "NonFiniteEvaluation");
}
