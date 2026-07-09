/**
 * @file numerics.hpp
 * @brief Umbrella header for the numerics module.
 *
 * The numerics module is intentionally **independent of finance**: nothing
 * inside `ore::numerics` may include headers from `ore::core`,
 * `ore::pricing`, `ore::analytics`, or any other domain module. This makes
 * numerics reusable, testable in isolation, and keeps the dependency
 * graph strictly one-way (finance -> numerics, never the reverse).
 *
 * Included by this umbrella (prefer including the concrete headers
 * directly in production code — the umbrella exists mostly for tests and
 * quick exploration):
 *
 *   - `constants.hpp`             — mathematical constants
 *   - `comparison.hpp`            — floating-point comparison helpers
 *   - `normal_distribution.hpp`   — `NormalDistribution` (PDF, CDF, inverse CDF)
 *   - `solver_result.hpp`         — `SolverStatus`, `SolverResult`
 *   - `newton_raphson.hpp`        — `NewtonRaphsonSolver`
 *   - `bisection.hpp`             — `BisectionSolver`
 *   - `random_number_generator.hpp` — `NormalGenerator` and the
 *                                   `MersenneTwisterNormalGenerator`
 *                                   (reproducible, seed-defaulted normal
 *                                   draws for the Monte Carlo engine).
 *
 * Planned but not yet implemented:
 *   - `interpolation.hpp` — 1D / 2D interpolation for vol surfaces.
 *   - `integration.hpp`   — numerical quadrature (only if needed).
 */
#pragma once

#include <ore/numerics/bisection.hpp>
#include <ore/numerics/comparison.hpp>
#include <ore/numerics/constants.hpp>
#include <ore/numerics/newton_raphson.hpp>
#include <ore/numerics/normal_distribution.hpp>
#include <ore/numerics/random_number_generator.hpp>
#include <ore/numerics/solver_result.hpp>
