"""GTA VI trophy earn-rate forecast (gta6-trophy-rarity-forecast-01).

Model: logit-trend + tier shrinkage.
For each trophy, fit a straight line to logit(rate) against
ln(1 + hours since 2026-11-19T00:00:00Z) over the case's reads, shrink
each slope halfway toward its tier's mean slope, read the line at the
target, then apply a mild dilution adjustment from projected growth in
players_tracked. Spread combines a random-walk term over the horizon
with the fit's residual variance. Flat histories fall back to the last
rate (slope zero), exactly as the house curve does on launch night.

Python 3.13, standard library only, deterministic, no network, no
clock reads, no randomness. Reads every *.in in MAKEFUN_INPUT_DIR
(default /input), writes <id>.out to MAKEFUN_OUTPUT_DIR (default
/output): {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
ORIGIN = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Random-walk spread, logit units per square-root day (house baseline's).
SIGMA_RW = 0.5
# How far a trophy's slope is pulled toward its tier mean (0 = own slope).
SHRINK = 0.5
# Dilution exponent: predicted rate scales by (players_last /
# players_target) ** DILUTION_EXP, capped to at most a halving.
DILUTION_EXP = 0.35


def clamp_pct(x):
    if x < 0.01:
        return 0.01
    if x > 99.99:
        return 99.99
    return x


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def hours_since_origin(iso):
    ts = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return max(0.0, (ts - ORIGIN).total_seconds() / 3600.0)


def fit_line(xs, ys):
    """Least-squares line y = a + b*x. Returns (a, b, resid_var)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n
    return a, b, resid


def project_players(history, horizon_minutes):
    """Project players_tracked at the target by log-linear growth.

    Returns the dilution multiplier for rates: (last / projected) **
    DILUTION_EXP, clamped to [0.5, 1.0]. Flat or shrinking counts give 1.
    """
    pts = []
    for read in history:
        n = read.get("players_tracked")
        if isinstance(n, (int, float)) and n and n > 0:
            pts.append((hours_since_origin(read["at"]), math.log(n)))
    if len(pts) < 2:
        return 1.0
    a, b, _ = fit_line([p[0] for p in pts], [p[1] for p in pts])
    if b <= 0:
        return 1.0
    last_h = pts[-1][0]
    target_h = last_h + horizon_minutes / 60.0
    growth = math.exp(b * (target_h - last_h))
    if growth <= 1.0:
        return 1.0
    return max(0.5, growth ** (-DILUTION_EXP))


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(name + " asks for levels " + repr(case["levels"]))
    history = case["history"]
    if not history:
        raise ValueError(name + " has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    target = datetime.fromisoformat(
        case["target"].replace("Z", "+00:00"))
    x_target = math.log1p(max(0.0, (target - ORIGIN).total_seconds() / 3600.0))
    dilution = project_players(history, case["horizon_minutes"])
    last = history[-1]

    # First pass: per-trophy line fits.
    fits = []
    for i, trophy in enumerate(trophies):
        xs, ys = [], []
        for read in history:
            rate = read["rates"][i]
            if rate is None:
                continue
            xs.append(math.log1p(hours_since_origin(read["at"])))
            ys.append(logit(rate))
        if len(xs) >= 3 and len(set(xs)) >= 2 and len(set(ys)) >= 2:
            a, b, resid = fit_line(xs, ys)
        else:
            base = last["rates"][i]
            a = logit(base if base is not None else 0.0)
            b, resid = 0.0, 0.0
        fits.append((a, b, resid, trophy["tier"]))

    # Tier mean slopes, for shrinkage.
    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        slopes = [f[1] for f in fits if f[3] == tier]
        tier_slopes[tier] = sum(slopes) / len(slopes) if slopes else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        a, b, resid, tier = fits[i]
        b_shrunk = (1.0 - SHRINK) * b + SHRINK * tier_slopes[tier]
        centre = a + b_shrunk * x_target
        # A saturated trophy (100% in the last read) stays saturated:
        # dilution cannot pull it down, the refresh left it at 100%.
        if last["rates"][i] is not None and last["rates"][i] >= 99.99:
            median = 99.99
        else:
            median = sigmoid(centre) * dilution
        # Keep the median inside (0, 100); dilution only ever lowers it.
        median = min(99.99, max(0.01, median))
        spread = math.sqrt((SIGMA_RW ** 2) * horizon_days + resid)
        # Near the 100% ceiling the band must stay inside the scale:
        # cap the spread so q05 stays at or above 99.0 for a 100% trophy.
        last_rate = last["rates"][i]
        if last_rate is not None and last_rate >= 99.99:
            cap = (logit(99.99) - logit(99.0)) / 1.645
            spread = min(spread, max(cap, 0.01))
            median = 100.0 if median > 99.99 else median
        med_logit = logit(median)
        qs = []
        for z in Z:
            v = sigmoid(med_logit + z * spread)
            v = min(100.0, max(0.0, v))
            qs.append(round4(v))
        # Enforce non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
