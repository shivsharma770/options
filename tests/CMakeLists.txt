# Test target for OptionResearchEngine.
#
# GoogleTest is pulled in via FetchContent so no system install is required
# and the version is reproducible.
#
# Tests are organised in subdirectories that mirror the include/ layout:
#   tests/core/       - ore::core value types
#   tests/io/         - ore::io CSV reader
#   tests/marketdata/ - ore::marketdata loaders
#   tests/numerics/   - ore::numerics utilities
#   tests/pricing/    - ore::pricing engines
#   tests/analytics/  - ore::analytics utilities
#   tests/benchmark/  - ore::benchmark framework
#   tests/utils/      - ore::utils utilities (empty for now)

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2  # Release-1.15.2 (pinned; bump deliberately)
)

# On Windows, force GoogleTest to use the same CRT (shared/DLL) as the rest
# of the project. Without this, MSVC will refuse to link.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Do not install GoogleTest as part of `cmake --install`.
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# --- Test executable --------------------------------------------------------
add_executable(ore_tests
    test_version.cpp

    core/test_option.cpp
    core/test_option_market_snapshot.cpp
    core/test_underlying.cpp
    core/test_option_chain.cpp

    io/test_csv_reader.cpp

    marketdata/test_yahoo_option_loader.cpp
    marketdata/test_historical_loader.cpp
    marketdata/test_eod_options_loader.cpp

    numerics/test_constants.cpp
    numerics/test_comparison.cpp
    numerics/test_normal_distribution.cpp
    numerics/test_newton_raphson.cpp
    numerics/test_bisection.cpp
    numerics/test_random_number_generator.cpp

    pricing/test_black_scholes.cpp
    pricing/test_binomial_tree_engine.cpp
    pricing/test_implied_volatility.cpp
    pricing/test_monte_carlo_engine.cpp

    analytics/test_option_chain_calibrator.cpp
    analytics/test_volatility_analytics.cpp

    benchmark/test_benchmark.cpp

    research/test_historical_dataset_extensions.cpp
    research/test_research_engine.cpp
    research/test_iv_validation_study.cpp
    research/test_greeks_validation_study.cpp
    research/test_pricing_validation_study.cpp
)

target_link_libraries(ore_tests PRIVATE
    ore::ore
    GTest::gtest_main
)

# Tests that load fixture CSVs need to know where they live. We inject the
# absolute path at compile time as a preprocessor macro. Prefer this to
# depending on the working directory (which CTest may or may not set to a
# predictable location).
target_compile_definitions(ore_tests PRIVATE
    ORE_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)

# Relaxed warnings (no /WX or -Werror) so any strict-warning noise that
# leaks from GoogleTest headers into our test TUs cannot break the build.
# Warnings in *our* code still surface as warnings and should be fixed.
ore_set_warnings_no_werror(ore_tests)

# CTest integration: discover every TEST() at build time.
include(GoogleTest)
gtest_discover_tests(ore_tests
    PROPERTIES
        LABELS "unit"
    DISCOVERY_TIMEOUT 30
)
