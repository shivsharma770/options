#include <ore/numerics/random_number_generator.hpp>

#include <cstdint>
#include <memory>

namespace ore::numerics {

MersenneTwisterNormalGenerator::MersenneTwisterNormalGenerator(std::uint64_t seed)
    : engine_(seed) {}

double MersenneTwisterNormalGenerator::next() {
    // `std::normal_distribution` is stateful — it may cache one of the
    // two values from each Box-Muller pair. The distribution object
    // therefore travels with the generator across calls and is included
    // in the deep-copy performed by `clone()`.
    return dist_(engine_);
}

void MersenneTwisterNormalGenerator::seed(std::uint64_t seed) {
    engine_.seed(seed);
    // `std::normal_distribution::reset()` discards any cached sample
    // from the last call to `operator()`. Without this, a seed reset
    // may deliver a stale value on the next `next()` call, which would
    // silently break common-random-number Greek estimation.
    dist_.reset();
}

std::unique_ptr<NormalGenerator> MersenneTwisterNormalGenerator::clone() const {
    // The copy-constructor of `std::mt19937_64` performs a full state
    // copy; `std::normal_distribution` copies its cached-value flag.
    // Together these produce a bit-identical independent generator.
    return std::make_unique<MersenneTwisterNormalGenerator>(*this);
}

} // namespace ore::numerics
