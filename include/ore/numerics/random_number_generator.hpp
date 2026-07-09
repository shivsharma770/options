/**
 * @file random_number_generator.hpp
 * @brief Reproducible standard-normal random number generation for
 *        stochastic pricing engines.
 *
 * ### Why this lives in `ore::numerics`
 *
 * Following the module's finance-independence rule: everything here
 * knows only about `double`, `std::uint64_t`, and the standard library.
 * The Monte Carlo pricer, and every future stochastic-simulation
 * pricer, depends on this header.
 *
 * ### Interface
 *
 * A minimal abstract base `NormalGenerator` exposes three operations:
 *
 *   * `next()` — draw a single N(0, 1) sample.
 *   * `seed(s)` — reset the internal state to a deterministic point.
 *   * `clone()` — deep-copy the generator (useful for parallel runs and
 *                 for common-random-number Greek bumps in the future).
 *
 * The initial concrete implementation is `MersenneTwisterNormalGenerator`,
 * which wraps `std::mt19937_64` + `std::normal_distribution<double>`.
 * Later milestones may add Sobol / Halton / Ziggurat variants; the
 * abstract interface is deliberately small so that adding them is a
 * pure implementation exercise — no consumer needs to change.
 *
 * ### Design trade-off (virtual vs template)
 *
 * We chose *virtual dispatch* over a compile-time policy because:
 *
 *   * `MonteCarloEngine` inherits from the non-template
 *     `PricingEngine`; templating it downstream forces every consumer
 *     to be templated too.
 *   * The overhead of one virtual call per sample is a couple of
 *     nanoseconds; on a million-path run that is ~2 ms of total
 *     dispatch cost — negligible compared with the ~50 ms actually
 *     spent in the `normal_distribution` transform and the payoff
 *     arithmetic.
 *   * Quasi-random sequences (Sobol, Halton) carry substantially
 *     different state than a Mersenne Twister; abstracting via a
 *     small virtual interface keeps that additional complexity out of
 *     the pricing engines.
 *
 * ### Thread-safety
 *
 * Instances are **not** thread-safe: a single generator holds mutable
 * state (the RNG's internal buffer). Give each thread its own generator,
 * or use `clone()` on a seeded template to make deterministic per-worker
 * copies.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string_view>

namespace ore::numerics {

/**
 * @brief Abstract producer of independent standard-normal samples.
 *
 * The interface is *strictly value-typed* — every derived class owns
 * its own state, and `clone()` returns a deep copy of that state. This
 * mirrors the value semantics of every other type in the numerics
 * module.
 */
class NormalGenerator {
public:
    virtual ~NormalGenerator() = default;

    /** Draw a single N(0, 1) sample. */
    [[nodiscard]] virtual double next() = 0;

    /** Reset the generator to a deterministic seed. Subsequent draws
     *  are the same as those from a freshly constructed instance with
     *  the same seed. */
    virtual void seed(std::uint64_t seed) = 0;

    /** Deep-copy the generator, including its current state. Useful
     *  for common-random-number Greek estimation (bump inputs, reuse
     *  seed) and for splitting work across threads deterministically. */
    [[nodiscard]] virtual std::unique_ptr<NormalGenerator> clone() const = 0;

    /** Short human-readable identifier — "MersenneTwister64", "Sobol",
     *  etc. Useful for diagnostics and reproducibility logs. */
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

protected:
    NormalGenerator() = default;
    NormalGenerator(const NormalGenerator&) = default;
    NormalGenerator(NormalGenerator&&) noexcept = default;
    NormalGenerator& operator=(const NormalGenerator&) = default;
    NormalGenerator& operator=(NormalGenerator&&) noexcept = default;
};

/**
 * @brief 64-bit Mersenne Twister + `std::normal_distribution` — the
 *        default generator for every stochastic pricer.
 *
 * Uses `std::mt19937_64` (period \f$2^{19937} - 1\f$, passes every
 * standard test suite for pseudo-randomness within its known
 * limitations) combined with the standard library's Box-Muller-based
 * `std::normal_distribution<double>` to produce N(0, 1) draws.
 *
 * @note The choice of `std::normal_distribution` transform is
 * implementation-defined across standard libraries — libc++, libstdc++,
 * and MSVC each use a slightly different flavour of Box-Muller /
 * Ziggurat. This means bit-identical reproducibility is only guaranteed
 * within a single build environment. That is acceptable for a research
 * engine; production reproducibility would require shipping our own
 * normal transform (an explicit later milestone if needed).
 */
class MersenneTwisterNormalGenerator final : public NormalGenerator {
public:
    /** Construct with an explicit seed. Default seed of 42 matches the
     *  Config default of `MonteCarloEngine`. */
    explicit MersenneTwisterNormalGenerator(std::uint64_t seed = 42);

    /** @copydoc NormalGenerator::next */
    [[nodiscard]] double next() override;

    /** @copydoc NormalGenerator::seed */
    void seed(std::uint64_t seed) override;

    /** @copydoc NormalGenerator::clone */
    [[nodiscard]] std::unique_ptr<NormalGenerator> clone() const override;

    /** @copydoc NormalGenerator::name */
    [[nodiscard]] std::string_view name() const noexcept override {
        return "MersenneTwister64";
    }

private:
    std::mt19937_64                  engine_;
    std::normal_distribution<double> dist_{0.0, 1.0};
};

} // namespace ore::numerics
