# data/

Local data cache. Populated by the Python downloader (Milestone 2+) and by
outputs of the C++ engine.

## Layout

```
data/
    raw/         # Downloaded, unmodified market data (from yfinance, Polygon, etc.)
    processed/   # Normalized, cleaned, ready-to-parse-by-C++ datasets
    generated/   # Outputs produced by the engine (vol surfaces, benchmark
                 # results, backtest logs, etc.)
```

Everything under `raw/`, `processed/`, and `generated/` is **gitignored**
apart from `.gitkeep` sentinels and small test fixtures explicitly checked
in. Do not commit market data into the repository.
