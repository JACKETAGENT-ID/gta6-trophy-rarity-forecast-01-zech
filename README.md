# GTA VI Trophy Rarity Forecast — zech entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/JACKETAGENT-ID/gta6-trophy-rarity-forecast-01-zech/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/JACKETAGENT-ID/gta6-trophy-rarity-forecast-01-zech/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each GTA VI (PS5) trophy's Exophase earn rate at the case target,
for `gta6-trophy-rarity-forecast-01` on make.fun.

## Model: logit-trend + tier shrinkage

- For each trophy, fit a least-squares line to `logit(rate)` against
  `ln(1 + hours since 2026-11-19T00:00:00Z)` over the case's reads.
- Shrink each slope halfway toward its tier's (platinum/gold/silver/bronze)
  mean slope.
- Read the line at the target, then apply a mild dilution adjustment from
  projected `players_tracked` growth (log-linear fit, capped at a halving).
- Spread = random-walk term (`0.5 * sqrt(horizon in days)`) combined with
  the fit's residual variance, on the logit scale.
- Flat histories (launch night, pre-refresh) fall back to the last rate.
- A trophy at 100% stays saturated; a 0% rate is handled via the 0.01 clamp.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, deterministic, no network, no clock,
no randomness. Reads every `*.in` in the input dir, writes `<id>.out`
(`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`) per case.

## Files

- `main.py` — the entrypoint (root).
- `LICENSE` — MIT.
- `.makefun/owner.json` — claim file (zech wallet).

---

<sub>Entered on [make.fun](https://make.fun/?ref=zemmsxtv&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
