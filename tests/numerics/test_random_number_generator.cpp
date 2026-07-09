/**
 * @file test_random_number_generator.cpp
 * @brief Correctness tests for `ore::numerics::MersenneTwisterNormalGenerator`.
 *
 * These tests focus on the *reproducibility and statistical* contract
 * of the generator, not on the quality of the Mersenne Twister itself
 * (which is a well-studied external component).
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <string_view>
#include <vector>

#include <ore/numerics/random_number_generator.hpp>

using ore::numerics::MersenneTwisterNormalGenerator;
using ore::numerics::NormalGenerator;

namespace {

std::vector<double> draw(NormalGenerator& g, std::size_t n) {
    std::vector<double> xs(n);
    for (std::size_t i = 0; i < n; ++i) xs[i] = g.next();
    return xs;
}

}  // namespace

// =============================================================================
// REPRODUCIBILITY
// =============================================================================

TEST(NormalGeneratorTest, SameSeedProducesIdenticalSequence) {
    MersenneTwisterNormalGenerator a(1234);
    MersenneTwisterNormalGenerator b(1234);
    const auto xs = draw(a, 1000);
    const auto ys = draw(b, 1000);
    for (std::size_t i = 0; i < xs.size(); ++i) {
        EXPECT_DOUBLE_EQ(xs[i], ys[i]) << "diverged at i=" << i;
    }
}

TEST(NormalGeneratorTest, DifferentSeedsProduceDifferentSequences) {
    MersenneTwisterNormalGenerator a(1);
    MersenneTwisterNormalGenerator b(2);
    const auto xs = draw(a, 100);
    const auto ys = draw(b, 100);
    // At least *some* elements must differ. (In principle collisions
    // are possible; with 100 draws the probability is effectively 0.)
    bool any_diff = false;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        if (xs[i] != ys[i]) { any_diff = true; break; }
    }
    EXPECT_TRUE(any_diff);
}

TEST(NormalGeneratorTest, ReseedResetsToInitialStream) {
    MersenneTwisterNormalGenerator g(42);
    const auto first = draw(g, 100);
    g.seed(42);
    const auto second = draw(g, 100);
    for (std::size_t i = 0; i < first.size(); ++i) {
        EXPECT_DOUBLE_EQ(first[i], second[i]);
    }
}

TEST(NormalGeneratorTest, ReseedDiscardsCachedNormalSample) {
    // Reseed after an *odd* number of draws — the Box-Muller/Ziggurat
    // transform in `std::normal_distribution` may have a cached second
    // half. Reset must discard it so the next draw is deterministic in
    // the new seed alone.
    MersenneTwisterNormalGenerator g(42);
    (void)g.next();  // consume one — leaves a cached partner in some impls
    g.seed(42);
    const double after_reseed = g.next();

    MersenneTwisterNormalGenerator fresh(42);
    const double from_fresh = fresh.next();
    EXPECT_DOUBLE_EQ(after_reseed, from_fresh);
}

// =============================================================================
// CLONE
// =============================================================================

TEST(NormalGeneratorTest, CloneProducesIndependentIdenticalSequence) {
    MersenneTwisterNormalGenerator g(7);
    (void)draw(g, 50);  // advance state
    auto c = g.clone();
    ASSERT_NE(c, nullptr);
    // From this point the original and the clone should agree.
    const auto x_orig = draw(g,    500);
    const auto x_clone = draw(*c,  500);
    for (std::size_t i = 0; i < x_orig.size(); ++i) {
        EXPECT_DOUBLE_EQ(x_orig[i], x_clone[i]);
    }
}

TEST(NormalGeneratorTest, CloneDoesNotShareState) {
    MersenneTwisterNormalGenerator g(11);
    auto c = g.clone();
    // Advance one but not the other.
    (void)draw(g, 100);
    const double from_clone = c->next();
    // The clone's next draw should equal what a fresh generator produces
    // after 0 prior draws — i.e. its own state, not `g`'s.
    MersenneTwisterNormalGenerator fresh(11);
    const double from_fresh = fresh.next();
    EXPECT_DOUBLE_EQ(from_clone, from_fresh);
}

// =============================================================================
// DISTRIBUTIONAL SANITY
// =============================================================================

TEST(NormalGeneratorTest, LargeSampleMatchesStandardNormalMoments) {
    // Central-limit-theorem sanity check. Sample size and tolerance are
    // set so the test never flakes with the fixed seed. Mean ≈ 0,
    // stddev ≈ 1 for N(0, 1).
    MersenneTwisterNormalGenerator g(2026);
    const std::size_t N = 200'000;
    const auto xs = draw(g, N);

    const double mean = std::accumulate(xs.begin(), xs.end(), 0.0)
                        / static_cast<double>(N);
    double sq_dev = 0.0;
    for (double x : xs) sq_dev += (x - mean) * (x - mean);
    const double sample_var = sq_dev / static_cast<double>(N - 1);
    const double sample_sd  = std::sqrt(sample_var);

    // Tolerance: for N=200k the sample mean is N(0, 1/sqrt(N)) → 1σ
    // ≈ 0.0022. We use 5σ to make the test comfortably deterministic.
    EXPECT_NEAR(mean,      0.0, 0.02);
    EXPECT_NEAR(sample_sd, 1.0, 0.02);
}

TEST(NormalGeneratorTest, LargeSampleHasApproximatelySymmetricTails) {
    // Count draws with |x| > 2. For N(0, 1), P(|X| > 2) ≈ 0.0455.
    MersenneTwisterNormalGenerator g(999);
    const std::size_t N = 200'000;
    std::size_t left = 0, right = 0;
    for (std::size_t i = 0; i < N; ++i) {
        const double x = g.next();
        if (x < -2.0) ++left;
        if (x >  2.0) ++right;
    }
    const double p_left  = static_cast<double>(left)  / static_cast<double>(N);
    const double p_right = static_cast<double>(right) / static_cast<double>(N);
    EXPECT_NEAR(p_left,  0.02275, 0.005);
    EXPECT_NEAR(p_right, 0.02275, 0.005);
}

// =============================================================================
// INTERFACE
// =============================================================================

TEST(NormalGeneratorTest, NameIsStableAndInformative) {
    MersenneTwisterNormalGenerator g;
    EXPECT_EQ(g.name(), std::string_view{"MersenneTwister64"});
}

TEST(NormalGeneratorTest, PolymorphicUseCompiles) {
    // Compile-time check that a `NormalGenerator*` interface actually
    // works for consumers. This is the whole point of the abstraction.
    std::unique_ptr<NormalGenerator> g =
        std::make_unique<MersenneTwisterNormalGenerator>(3);
    const double x = g->next();
    EXPECT_TRUE(std::isfinite(x));
}
