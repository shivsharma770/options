/**
 * @file research_test_helpers.hpp
 * @brief Small helpers shared by the research-module tests.
 *
 * The research studies want realistic *data-shaped* input: an
 * `OptionChain` with BS-consistent prices and provider Greeks. Every
 * test file re-uses the same synthetic generator here to avoid
 * copy-pasting 40 lines of setup.
 */
#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <ore/core/market_snapshot.hpp>
#include <ore/core/option.hpp>
#include <ore/core/option_chain.hpp>
#include <ore/core/option_market_snapshot.hpp>
#include <ore/core/provider_greeks.hpp>
#include <ore/core/quote.hpp>
#include <ore/core/types.hpp>
#include <ore/core/underlying.hpp>
#include <ore/marketdata/historical_dataset.hpp>
#include <ore/marketdata/historical_snapshot.hpp>
#include <ore/pricing/black_scholes_engine.hpp>

namespace ore::research::testing {

/// Fabricate a `HistoricalSnapshot` whose contracts are priced to be
/// exactly BS-consistent at a given `iv`. `mid = BS(iv)` so the IV
/// solver will recover `iv` back to solver tolerance, and BS Greeks
/// scaled into vendor units land exactly on the "provider" Greeks
/// we attach.
///
/// Produces `2 * strikes.size()` contracts (one call, one put per
/// strike).
inline ore::marketdata::HistoricalSnapshot make_synthetic_snapshot(
    std::chrono::year_month_day date,
    std::chrono::year_month_day expiration,
    double spot,
    double iv,
    double risk_free_rate,
    double dividend_yield,
    std::vector<double> strikes)
{
    using namespace ore::core;
    using namespace ore::pricing;

    const Underlying u{"SPY", "NYSE", AssetType::ETF};

    MarketSnapshot market{};
    market.spot           = spot;
    market.risk_free_rate = risk_free_rate;
    market.dividend_yield = dividend_yield;
    market.volatility     = iv;
    market.valuation_date = date;

    BlackScholesEngine engine{};
    std::vector<OptionMarketSnapshot> options;
    options.reserve(2u * strikes.size());

    for (double K : strikes) {
        for (auto side : {OptionType::Call, OptionType::Put}) {
            Option opt{};
            opt.underlying = u;
            opt.strike     = K;
            opt.expiration = expiration;
            opt.type       = side;
            opt.exercise   = ExerciseStyle::European;

            const auto pr = engine.price(opt, market);

            Quote q{};
            // A tight ±0.01 spread around the theoretical price is
            // enough to exercise the IV solver without introducing
            // any mid-vs-theoretical bias.
            q.bid = pr.price - 0.01;
            q.ask = pr.price + 0.01;
            if (q.bid < 0.0) q.bid = 0.0;
            q.last = pr.price;
            q.volume = 100.0;
            q.implied_volatility = iv;

            ProviderGreeks pg{};
            pg.delta = pr.greeks.delta;
            pg.gamma = pr.greeks.gamma;
            // Vendor units: theta/day, vega per-1%.
            pg.theta = pr.greeks.theta * (1.0 / 365.0);
            pg.vega  = pr.greeks.vega  * 0.01;
            pg.rho   = pr.greeks.rho;
            q.provider_greeks = pg;

            OptionMarketSnapshot oms{};
            oms.option = opt;
            oms.quote  = q;
            options.push_back(std::move(oms));
        }
    }

    OptionChain chain{u, market, std::move(options)};
    return ore::marketdata::HistoricalSnapshot{date, std::move(chain)};
}

/// Build a small `HistoricalDataset` of `days` consecutive weekdays,
/// each with a BS-consistent chain at IV `iv`. Used by the engine
/// tests that need a real dataset with predictable content.
inline ore::marketdata::HistoricalDataset make_synthetic_dataset(
    std::size_t days,
    std::chrono::year_month_day start = std::chrono::year{2024}/std::chrono::January/2,
    double iv = 0.20)
{
    std::vector<ore::marketdata::HistoricalSnapshot> snaps;
    snaps.reserve(days);
    auto sd = std::chrono::sys_days{start};
    for (std::size_t i = 0; i < days; ++i) {
        std::chrono::year_month_day day{sd};
        // Expiration one month out — a realistic short-dated smile.
        auto expiration = std::chrono::year_month_day{sd + std::chrono::days{30}};
        snaps.push_back(make_synthetic_snapshot(
            day, expiration,
            /*spot=*/400.0, iv,
            /*r=*/0.03, /*q=*/0.01,
            {390.0, 395.0, 400.0, 405.0, 410.0}));
        sd += std::chrono::days{1};
    }
    return ore::marketdata::HistoricalDataset{"SPY", std::move(snaps)};
}

} // namespace ore::research::testing
