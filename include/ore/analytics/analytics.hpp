/**
 * @file analytics.hpp
 * @brief Umbrella header for the analytics module.
 *
 * `ore::analytics` turns a market-data `OptionChain` into calibrated
 * implied-volatility diagnostics and the classical volatility-structure
 * representations. It sits above `ore::pricing` / `ore::numerics` and is
 * consumed by `ore::research` (see `HistoricalCalibrationStudy`, which
 * drives everything here across an entire historical archive).
 *
 * Included by this umbrella (prefer including the concrete headers
 * directly in production code — the umbrella exists mostly for tests and
 * quick exploration):
 *
 *   - `option_chain_calibrator.hpp` — `OptionChainCalibrator`: solves an
 *     implied vol for every quotable contract in a chain, classifying the
 *     rest with a `SkipReason`. Produces a `CalibrationReport`.
 *   - `volatility_analytics.hpp`    — free functions over a
 *     `CalibrationReport`: `build_smiles`, `build_term_structure`,
 *     `build_surface`, `compute_skew_metrics`, plus IV summary
 *     statistics. All emit long-format CSV that the `python/plot_*.py`
 *     scripts consume directly.
 *
 * Design note: these operations are pure transformations over a
 * `CalibrationReport` with no per-run state, so the API is free functions
 * rather than classes. Deliberately absent (and owned by later work):
 * surface interpolation / parametric fitting, portfolio-level risk, and a
 * dedicated Greeks surface.
 *
 * The module still expects to sub-organise as it grows (e.g.
 * `analytics/statistics/`, `analytics/risk/`, `analytics/greeks/`); the
 * two headers above stay at the top level until a second file in one of
 * those areas justifies the split.
 */
#pragma once

#include <ore/analytics/option_chain_calibrator.hpp>
#include <ore/analytics/volatility_analytics.hpp>
