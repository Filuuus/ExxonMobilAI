# TxSON Soil Moisture Forecasting — Session Context & Implementation Plan

Last updated: 2026-09-02

## Project purpose

Ingest OpenMeteo weather + modeled soil moisture data to train a recursive LSTM that,
given **any coordinate**, forecasts a full year of volumetric soil moisture forward from
today. TxSON in-situ station data (TXS01-06 plus a wider CB/FD sensor network) is the
**ground-truth reference** used to validate forecast accuracy — it is not itself the thing
being deployed. The 2020-09-01 train / 2020-09-01→2021-09-01 blind-test split that shows
up in `txson_trend_optimized_pipeline.py` exists only because that's the historical window
where real ground truth is available to score against; production usage rolls forward from
the current date instead.

## Problem identified this session

`txson_trend_optimized_pipeline.py` (the current best pipeline: per-station baseline
z-score normalization + shared LSTM + Huber loss + 365-day zero-leakage recursive blind
validation) trains on `datasets/OpenMeteo_Synthetic_Grid_Dataset.csv`. That file only ever
contained OpenMeteo-sourced weather + soil-moisture data for the 6 original TXS01-06
coordinates (plus an unrelated Tamaulipas interpolation grid) — see
`openmeteo_historical_miner.py`'s `get_map_grid_coordinates()`.

Validation results (`txson_trend_optimized_metrics.csv`,
`txson_trend_optimized_validation.png`) confirmed the consequence:

| Group | R² | Corr | RMSE |
|---|---|---|---|
| TXS01–06 (had OpenMeteo training coverage) | 0.36 – 0.53 | 0.73 – 0.79 | ~0.04 |
| CB/FD stations (18 in that run, no OpenMeteo coverage) | -0.76 to -6.15 | ~0 to -0.33 | 0.07 – 0.17 |

For CB/FD stations the model collapsed to a smooth seasonal-only curve (driven by the
`DOY_sin/DOY_cos` features) with near-zero correlation to actual precipitation-driven
dynamics — it never saw location-appropriate OpenMeteo weather/moisture coupling for those
locations during training.

**Root cause:** spatial gap in training data, not a modeling flaw. Fix: fetch real
OpenMeteo weather + soil moisture at the actual CB/FD station coordinates, for their
relevant timeframes, and retrain.

## Data frequency verification (confirmed, no change needed)

Checked that CB/FD ingestion would match the TXS01-06 approach exactly:
- OpenMeteo `daily` fields return one row per calendar day; `hourly.soil_moisture_3_to_9cm`
  is aggregated to a daily mean. Confirmed against a live cached response.
- `look_back=14` used everywhere (`txson_trend_optimized_pipeline.py`,
  `forecaster_engine_v2.py`, etc.) is the LSTM's rolling **input window length**, not a
  data-sampling interval — the underlying data and the 365-day validation plot are both
  daily-resolution throughout.
- The only place a weekly stride appears is the 7-day animation-frame scrubber in
  `spatial_timeline_map_v2.py`'s heatmap — a visualization choice, unrelated to training
  data collection.

## Work completed this session

### New files
- **`openmeteo_cbfd_miner.py`** — parses CB/FD station coordinates from
  `datasets/final_clean/CSV/CB_FD_Station_info.txt`, cross-matches them against
  `datasets/final_clean/CSV/txson_expanded_station_catalog.csv` for valid per-station date
  ranges (clipped to `[2014-09-01, 2024-09-01]` to match the original miner's span), and
  fetches OpenMeteo archive weather + soil moisture at each real coordinate. Also
  regenerates the TXS01-06 anchor + spatial grid + Tamaulipas grid (relabeling the 6 real
  TxSON anchors from `GRID_TX_A_TXS0X` to plain `TXS0X` so they line up with real in-situ
  station names for blind validation). Writes the unified result to
  `datasets/OpenMeteo_Synthetic_Grid_Dataset.csv` (same filename
  `txson_trend_optimized_pipeline.py` already expects — no pipeline changes needed to
  consume it).
  - 23 of 27 catalog-valid CB/FD stations had harvestable coordinates and were matched.
  - **Missing coordinates (skipped, not in the harvested info file):** `CB07`, `CB15`,
    `FD14`, `FD24`. `CB15` has no usable data anyway (0 hours in catalog); `CB07`, `FD14`,
    `FD24` do have real data but no coordinates yet — if the user finds/adds them to
    `CB_FD_Station_info.txt`, rerunning the miner will pick them up automatically.
- **`spatial_timeline_map_v3.py`** — the "any coordinate → 1 year forward forecast" map
  script, built on the (about to be retrained) trend-optimized model rather than the older
  `expanded_lstm_model.keras` raw-value model. Key design points, see "Design decisions"
  below. **Not yet tested** — depends on artifacts that don't exist until the pipeline is
  rerun (see Implementation Plan).

### Modified files
- **`txson_trend_optimized_pipeline.py`**: now persists the fitted weather `MinMaxScaler`
  via `joblib.dump(scaler_weather, 'trend_optimized_weather_scaler.joblib')` right after
  fitting it. It was previously kept in memory only, which meant a separate inference
  script (`spatial_timeline_map_v3.py`) couldn't reproduce identical scaling. Verified (via
  a background check) that this did not touch the per-station z-score normalization logic —
  baselines are still computed per `Station_ID`, and the shared `scaler_weather` only scales
  meteorological *inputs*, never the soil-moisture target.
- **`openmeteo_historical_miner.py`**:
  - `fetch_historical_station_with_retry`: retries raised 3 → 8, backoff changed from linear
    `(attempt+1)*15s` to exponential `min(180, 20 * 2**attempt)` — the original scheme
    wasn't long enough to survive a sustained rate-limit window.
  - Added `fetch_historical_station_cached` (disk-cached wrapper): caches each successful
    per-station fetch to `.cache/openmeteo_miner/{Station_ID}_{start}_{end}.csv` and returns
    `(df, was_cached)`. Makes the whole mining process resumable — an interrupted run no
    longer has to re-fetch (and re-burn rate-limit budget on) stations it already completed.
  - Inter-request delay raised 3.5s → 6.0s, and skipped entirely on cache hits.
  - `build_openmeteo_dataset()` updated to use the cached fetcher and report `[cached]` hits.

## Design decisions

1. **Map script uses the trend-optimized (baseline-normalized) model**, not the older raw
   `expanded_lstm_model.keras`. Rationale, confirmed with the user:
   - Train/inference consistency: trend-optimized trains entirely on OpenMeteo data, and the
     map script also seeds from live OpenMeteo data at inference time — same distribution.
     The old raw model trains on real sensor values (`TxSON_Expanded_Unified_Dataset.csv`)
     but gets fed OpenMeteo data at inference — a domain mismatch.
   - Baseline interpolation is now far more reliable: ~29 known-baseline points (6 TXS +
     23 CB/FD, plus the synthetic TX/Tamaulipas grid) instead of just 6 anchors.
   - It's the model actively being validated/fixed this session — shipping the map on a
     different, unvalidated architecture would decouple the accuracy work from what's shown.
2. **IDW baseline interpolation** (`interpolate_baseline` in `spatial_timeline_map_v3.py`):
   for an arbitrary query coordinate with no known station, `(mean, std)` is estimated via
   inverse-distance weighting (power=2, k=6 nearest) over every known-baseline
   station/grid point (`station_soil_baselines.json` keys, matched to coordinates from
   `get_map_grid_coordinates()` + `parse_station_coordinates()`).
3. **Forward rollout, not historical backtest**: `spatial_timeline_map_v3.py` seeds from a
   live 14-day OpenMeteo window ending today and rolls 364 days forward, matching the
   deployment behavior the user described (unlike v2, which rolled forward from whatever
   date the script happened to run and used a synthetic, arbitrarily-phased sine wave for
   seasonal forcing).
4. **DOY-exact seasonal forcing**: future `T_Max`/`T_Min` forcing during rollout comes from
   a day-of-year climatology table built from the real historical OpenMeteo training data
   (`build_doy_climatology`), and `DOY_sin`/`DOY_cos` are computed from real future calendar
   dates — more accurate than v2's guessed sinusoid, and consistent with what the model
   actually learned (DOY_sin/cos are explicit trained features in the trend-optimized
   architecture). Precipitation forcing remains a conservative zero/drydown default, same
   convention as v2.
5. CB/FD real station coordinates are added as explicit labeled points on the map grid
   (`"CB/FD Station ({sid})"`) alongside the existing TX/Tamaulipas synthetic grid, so the
   heatmap visibly reflects the expanded network.

## Current status (as of this file being written)

- `openmeteo_cbfd_miner.py` was run once and **failed partway through**: the 79-coordinate
  TXS/Tamaulipas grid phase succeeded (after several 429 retries), but the CB/FD phase then
  hit sustained rate-limiting from the first station onward and started skipping stations
  outright (old retry logic was too weak to recover).
- The retry/backoff/caching fixes above were made **after** that failed run, so the 79
  already-completed grid fetches from that run were not cached anywhere and will need to be
  re-fetched from scratch on restart. From this point forward, every successful fetch
  persists to `.cache/openmeteo_miner/`.
- **`datasets/OpenMeteo_Synthetic_Grid_Dataset.csv` does not exist yet.**
- Neither `txson_trend_optimized_pipeline.py` nor `spatial_timeline_map_v3.py` has been
  rerun/tested against the expanded data yet.

## Implementation plan / next steps

1. **Restart the miner**: `python openmeteo_cbfd_miner.py`. With the new backoff/pacing/
   caching it should either complete cleanly or, if interrupted again, resume from cache on
   the next run instead of losing progress. Expect ~10-15 minutes for a full clean run
   (102 coordinates × ~6-10s each, plus any retry cooldowns).
2. **Sanity-check the output** before training: row counts and date coverage per
   `Station_ID` in `datasets/OpenMeteo_Synthetic_Grid_Dataset.csv`, confirm all 23 CB/FD
   stations plus TXS01-06 plus the grid points are present with no all-NaN columns.
3. **Rerun `txson_trend_optimized_pipeline.py`** against the expanded dataset. This is the
   actual fix-verification step — check that CB/FD stations with sufficient train+test
   coverage (same ≥350-test-day / std>0.005 filter as before) now show materially better
   R²/Corr, not just a flat seasonal curve. Note: short-lived stations (FD08, FD16, FD21,
   FD22 — all post-2020 starts) will still be excluded from blind validation by that filter,
   same as before; that's expected, not a bug.
   - This also produces `trend_optimized_weather_scaler.joblib` and an updated
     `station_soil_baselines.json` (now covering ~29 stations instead of 6), both required
     by `spatial_timeline_map_v3.py`.
4. **Run and visually test `spatial_timeline_map_v3.py`**: confirm the interpolated
   baselines look physically reasonable across the region (no wild extrapolation at the
   basin edges), confirm the live OpenMeteo seed fetch + forward rollout completes, and
   inspect the resulting `capstone_spatial_forecast_v3.html` heatmap for sane spatial
   coherence (moisture gradients should track known wetter/drier station baselines, not
   look noisy/discontinuous between grid cells and CB/FD points).
5. **Open item**: `CB07`, `FD14`, `FD24` remain untrained due to missing coordinates. If the
   user locates them, add to `CB_FD_Station_info.txt` in the same format and rerun the
   miner — no code changes needed, they'll be picked up automatically by
   `build_cbfd_targets()`.
