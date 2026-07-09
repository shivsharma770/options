# Main library target for OptionResearchEngine.
#
# Rationale
# ---------
# Still a single STATIC library `ore` aggregating every module. Each .cpp
# file lives in the appropriate module subdirectory (`io/`, `marketdata/`,
# `numerics/`, ...); the flat source list here stays readable at this
# scale. If any module later grows more than ~5 files or gains its own
# external dependency, split it into its own CMake target.

add_library(ore STATIC
    version.cpp

    io/csv_reader.cpp

    marketdata/yahoo_option_loader.cpp
    marketdata/historical_dataset.cpp
    marketdata/historical_loader.cpp
    marketdata/eod_options_loader.cpp

    numerics/normal_distribution.cpp
    numerics/random_number_generator.cpp

    pricing/black_scholes_engine.cpp
    pricing/binomial_tree_engine.cpp
    pricing/implied_volatility_solver.cpp
    pricing/monte_carlo_engine.cpp

    analytics/option_chain_calibrator.cpp
    analytics/volatility_analytics.cpp

    benchmark/benchmark_case.cpp
    benchmark/benchmark_runner.cpp

    research/historical_research_engine.cpp
    research/research_report.cpp
    research/research_csv.cpp
    research/iv_validation_study.cpp
    research/greeks_validation_study.cpp
    research/pricing_validation_study.cpp
    research/historical_calibration_study.cpp
)

add_library(ore::ore ALIAS ore)

target_include_directories(ore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(ore PUBLIC cxx_std_20)

# The research module uses `std::thread` — link the FindThreads target so
# POSIX platforms pull in `-pthread`. On MSVC this is a no-op.
target_link_libraries(ore PUBLIC Threads::Threads)

ore_set_warnings(ore)
