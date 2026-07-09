#include <gtest/gtest.h>

#include <ore/numerics/newton_raphson.hpp>
#include <ore/numerics/solver_result.hpp>

#include <cmath>
#include <limits>

using ore::numerics::NewtonRaphsonSolver;
using ore::numerics::SolverResult;
using ore::numerics::SolverStatus;

// ---- Textbook problems ----------------------------------------------------

TEST(NewtonRaphsonTest, FindsSquareRootOfTwo) {
    // Convergence is on the *residual* |f(x)| < tolerance, so to assert a
    // 1e-12 error on the *root* the residual tolerance must be tighter than
    // 1e-12 * |f'(root)| = 1e-12 * 2*sqrt(2) ~ 2.8e-12. The default 1e-10
    // stops one Newton step early (root error ~1.6e-12); 1e-13 lets the
    // quadratic iteration take that final step.
    const NewtonRaphsonSolver solver{{.tolerance = 1e-13}};
    const auto f      = [](double x) { return x * x - 2.0; };
    const auto fprime = [](double x) { return 2.0 * x; };

    const SolverResult r = solver.solve(f, fprime, 1.0);
    ASSERT_TRUE(r.converged()) << to_string(r.status);
    EXPECT_NEAR(r.root, std::sqrt(2.0), 1e-12);
    EXPECT_LT(r.iterations, 15U);
}

TEST(NewtonRaphsonTest, FindsCubicRoot) {
    // f(x) = x^3 - x - 2, real root at ~1.5213797068045676
    const NewtonRaphsonSolver solver;
    const auto f      = [](double x) { return x * x * x - x - 2.0; };
    const auto fprime = [](double x) { return 3.0 * x * x - 1.0; };

    const SolverResult r = solver.solve(f, fprime, 1.5);
    ASSERT_TRUE(r.converged());
    EXPECT_NEAR(r.root, 1.5213797068045676, 1e-12);
}

TEST(NewtonRaphsonTest, FindsFixedPointOfCosine) {
    // Dottie's number: root of cos(x) - x = 0, at ~0.7390851332151607.
    const NewtonRaphsonSolver solver;
    const auto f      = [](double x) { return std::cos(x) - x; };
    const auto fprime = [](double x) { return -std::sin(x) - 1.0; };

    const SolverResult r = solver.solve(f, fprime, 0.5);
    ASSERT_TRUE(r.converged());
    EXPECT_NEAR(r.root, 0.7390851332151607, 1e-12);
}

// ---- Precision & performance ---------------------------------------------

TEST(NewtonRaphsonTest, IterationsShrinkResidualQuadratically) {
    // Newton on sqrt(2) roughly doubles correct digits each step. Two
    // iterations from a good guess should already be at ~1e-6.
    NewtonRaphsonSolver solver({.tolerance = 1e-100,   // never satisfied
                                .max_iterations = 2,
                                .min_derivative = 1e-14});
    const auto f      = [](double x) { return x * x - 2.0; };
    const auto fprime = [](double x) { return 2.0 * x; };
    const auto r      = solver.solve(f, fprime, 1.4);
    EXPECT_EQ(r.status, SolverStatus::MaxIterationsReached);
    EXPECT_NEAR(r.root, std::sqrt(2.0), 1e-6);
    EXPECT_EQ(r.iterations, 2U);
}

TEST(NewtonRaphsonTest, HonoursCustomTolerance) {
    NewtonRaphsonSolver strict({.tolerance = 1e-14});
    const auto f      = [](double x) { return x * x - 2.0; };
    const auto fprime = [](double x) { return 2.0 * x; };
    const auto r      = strict.solve(f, fprime, 1.0);
    ASSERT_TRUE(r.converged());
    EXPECT_LT(r.residual, 1e-14);
}

// ---- Failure modes --------------------------------------------------------

TEST(NewtonRaphsonTest, DoesNotConvergeWhenNoRealRootExists) {
    // f(x) = x^2 + 1 has no real root. Newton wanders — sometimes it
    // oscillates until max_iterations, sometimes an iterate lands close
    // enough to zero that the derivative floor fires. Both are legitimate
    // non-convergence outcomes for this ill-posed problem.
    NewtonRaphsonSolver solver({.tolerance = 1e-10, .max_iterations = 30});
    const auto f      = [](double x) { return x * x + 1.0; };
    const auto fprime = [](double x) { return 2.0 * x; };
    const auto r      = solver.solve(f, fprime, 1.5);
    EXPECT_FALSE(r.converged());
    EXPECT_TRUE(r.status == SolverStatus::MaxIterationsReached ||
                r.status == SolverStatus::DerivativeTooSmall)
        << "unexpected status: " << to_string(r.status);
}

TEST(NewtonRaphsonTest, ReportsDerivativeTooSmall) {
    // f(x) = x^3, f'(x) = 3x^2. At x0 = 0 the derivative is exactly zero.
    NewtonRaphsonSolver solver;
    const auto f      = [](double x) { return x * x * x; };
    const auto fprime = [](double x) { return 3.0 * x * x; };
    const auto r      = solver.solve(f, fprime, 0.0);
    // At x=0, f(x)=0 exactly, so we converge immediately without stepping
    // — the derivative floor is only tripped when f isn't already tiny.
    EXPECT_TRUE(r.converged());

    // Same problem starting a hair off zero triggers the derivative floor
    // before convergence.
    NewtonRaphsonSolver strict({.tolerance = 1e-16,
                                .max_iterations = 100,
                                .min_derivative = 1e-8});
    const auto r2 = strict.solve(f, fprime, 1e-3);
    EXPECT_EQ(r2.status, SolverStatus::DerivativeTooSmall);
}

TEST(NewtonRaphsonTest, ReportsNonFiniteEvaluation) {
    NewtonRaphsonSolver solver;
    const auto f      = [](double) { return std::numeric_limits<double>::quiet_NaN(); };
    const auto fprime = [](double x) { return 2.0 * x; };
    const auto r = solver.solve(f, fprime, 1.0);
    EXPECT_EQ(r.status, SolverStatus::NonFiniteEvaluation);
}

TEST(NewtonRaphsonTest, ReportsNonFiniteDerivative) {
    NewtonRaphsonSolver solver;
    const auto f      = [](double x) { return x * x - 2.0; };
    const auto fprime = [](double) { return std::numeric_limits<double>::infinity(); };
    const auto r = solver.solve(f, fprime, 1.0);
    EXPECT_EQ(r.status, SolverStatus::NonFiniteEvaluation);
}

// ---- Preconditions (must throw) ------------------------------------------

TEST(NewtonRaphsonTest, ThrowsOnNonFiniteInitialGuess) {
    NewtonRaphsonSolver solver;
    const auto f      = [](double x) { return x; };
    const auto fprime = [](double)    { return 1.0; };
    EXPECT_THROW(solver.solve(f, fprime, std::numeric_limits<double>::quiet_NaN()),
                 std::invalid_argument);
    EXPECT_THROW(solver.solve(f, fprime, std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
}

TEST(NewtonRaphsonTest, ThrowsOnNonPositiveTolerance) {
    NewtonRaphsonSolver solver({.tolerance = 0.0});
    const auto f      = [](double x) { return x; };
    const auto fprime = [](double)    { return 1.0; };
    EXPECT_THROW(solver.solve(f, fprime, 1.0), std::invalid_argument);
}

TEST(NewtonRaphsonTest, ThrowsOnZeroMaxIterations) {
    NewtonRaphsonSolver solver({.tolerance = 1e-10, .max_iterations = 0});
    const auto f      = [](double x) { return x; };
    const auto fprime = [](double)    { return 1.0; };
    EXPECT_THROW(solver.solve(f, fprime, 1.0), std::invalid_argument);
}

TEST(NewtonRaphsonTest, ThrowsOnNegativeMinDerivative) {
    NewtonRaphsonSolver solver({.tolerance = 1e-10,
                                .max_iterations = 100,
                                .min_derivative = -1.0});
    const auto f      = [](double x) { return x; };
    const auto fprime = [](double)    { return 1.0; };
    EXPECT_THROW(solver.solve(f, fprime, 1.0), std::invalid_argument);
}

// ---- Config accessor ------------------------------------------------------

TEST(NewtonRaphsonTest, ConfigIsExposed) {
    NewtonRaphsonSolver solver({.tolerance = 1e-8, .max_iterations = 42});
    EXPECT_EQ(solver.config().tolerance, 1e-8);
    EXPECT_EQ(solver.config().max_iterations, 42U);
}
