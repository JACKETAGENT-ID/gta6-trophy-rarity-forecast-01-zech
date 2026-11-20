"""GTA VI trophy earn-rate forecast (gta6-trophy-rarity-forecast-01), round 2.

Two regimes, picked from the case's own history:

1. FLAT history (every read lists the same rates -- launch night before
   Exophase's first refresh of the game): the next refresh does NOT leave
   the rates where they are. Exophase's tracked base jumps several-fold
   (the RULES' own account: 638 players tracked through launch night,
   3,868 at the first refresh, ~6x) and the newcomers have barely played,
   so they dilute every rate down -- except the one trophy every tracked
   player holds (100%), which stays at 100%. The median models that
   refresh explicitly:
       median_i = r_i * (1 + (G - 1) * beta_i) / G
   with G = 6 the refresh growth prior and beta_i the share of newcomers
   who have earned trophy i. beta rises steeply with the trophy's current
   rate -- newcomers earn the earliest trophies too, almost never the
   hard ones: beta_i = min(1, 0.05 * r_i ** 3.2). The pair (G, beta) is
   calibrated so the house baseline's published practice aggregates
   (pinball loss 1.508366, typical miss 0.68 pp on p00/p01) are
   reproduced by the baseline forecast. Round 1 missed this entirely:
   it forecast persistence with a 0.125-logit band, so the refresh took
   every rate outside its 90% band (coverage 0, loss 1.51).
   Spread here is a fixed refresh-magnitude term (0.35 logit) plus the
   random-walk term over the horizon, so the 90% band is wide enough to
   cover a refresh of somewhat different size.

2. TRENDING history (rates move between reads -- the final case): a
   least-squares line of logit(rate) against ln(1 + hours since release)
   over the case's last 24 hours of reads, slope shrunk halfway toward
   the tier mean, read at the target. No extra players-dilution factor:
   the observed trend already nets earning against dilution, and
   round 1 double-counted it. The platinum slope is damped x0.25 and
   its median capped at 2% (it needs every other trophy first). Spread
   combines the random-walk term, the fit's residual variance and the
   slope's sampling variance at the (extrapolated) target, capped at
   1.5 logit.

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
# Platinum slope damping: the platinum needs every other trophy first,
# so its rate cannot chase the pack.
PLATINUM_DAMP = 0.25
PLATINUM_CAP = 2.0

# --- First-refresh (flat-history) regime -------------------------------
# Tracked-base multiplier at the first refresh after launch.
REFRESH_GROWTH = 6.0
# Newcomer earn-share law: beta_i = min(1, BETA_C * r_i ** BETA_GAMMA).
BETA_C = 0.05
BETA_GAMMA = 3.2
# Fixed logit spread for the refresh magnitude itself.
SIGMA_REFRESH = 0.35

# --- Trend (moving-history) regime -------------------------------------
# Fit window: only the last 24 hours of reads (house curve's window).
TREND_WINDOW_HOURS = 24.0
SPREAD_CAP = 1.5
SPREAD_FLOOR = 0.15


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
    """Least-squares line y = a + b*x.

    Returns (a, b, resid_var, sxx, n, xbar); resid_var is the mean
    squared residual. sxx == 0 means x never moved (flat history).
    """
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.0, 0.0, n, mx
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n
    return a, b, resid, sxx, n, mx


def history_is_flat(history):
    """True when every read lists the same rates as the last read."""
    last_rates = history[-1]["rates"]
    for read in history:
        if read["rates"] != last_rates:
            return False
    return True


def last_known_rate(history, i):
    """Most recent non-null rate for trophy index i, else None."""
    for read in reversed(history):
        rate = read["rates"][i]
        if rate is not None:
            return rate
    return None


def refresh_median(rate):
    """First-refresh median for a trophy last seen at `rate` percent."""
    if rate is None:
        return 0.01
    if rate >= 99.99:
        return 100.0
    beta = min(1.0, BETA_C * (rate ** BETA_GAMMA))
    return rate * (1.0 + (REFRESH_GROWTH - 1.0) * beta) / REFRESH_GROWTH


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
    last = history[-1]
    flat = history_is_flat(history)

    # Trend fits (only used when the history moves). Window: last 24h.
    fits = []
    if not flat:
        cutoff = hours_since_origin(last["at"]) - TREND_WINDOW_HOURS
        for i, trophy in enumerate(trophies):
            xs, ys = [], []
            for read in history:
                if hours_since_origin(read["at"]) < cutoff:
                    continue
                rate = read["rates"][i]
                if rate is None:
                    continue
                xs.append(math.log1p(hours_since_origin(read["at"])))
                ys.append(logit(rate))
            if len(xs) >= 3 and len(set(xs)) >= 2 and len(set(ys)) >= 2:
                a, b, resid, sxx, n, xbar = fit_line(xs, ys)
            else:
                base = last_known_rate(history, i)
                a = logit(base if base is not None else 0.0)
                b, resid, sxx, n, xbar = 0.0, 0.0, 0.0, 0, 0.0
            fits.append((a, b, resid, sxx, n, xbar, trophy["tier"]))
        tier_slopes = {}
        for tier in ("platinum", "gold", "silver", "bronze"):
            slopes = [f[1] for f in fits if f[6] == tier]
            tier_slopes[tier] = sum(slopes) / len(slopes) if slopes else 0.0
    else:
        fits = [None] * len(trophies)
        tier_slopes = {}

    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last_known_rate(history, i)
        saturated = last_rate is not None and last_rate >= 99.99
        if flat:
            median = refresh_median(last_rate)
            spread = math.sqrt(
                SIGMA_REFRESH ** 2 + (SIGMA_RW ** 2) * horizon_days)
        else:
            a, b, resid, sxx, n, xbar, tier = fits[i]
            b_shrunk = (1.0 - SHRINK) * b + SHRINK * tier_slopes[tier]
            if tier == "platinum":
                b_shrunk *= PLATINUM_DAMP
            centre = a + b_shrunk * x_target
            if saturated:
                median = 99.99
            else:
                median = sigmoid(centre)
                if tier == "platinum":
                    median = min(median, PLATINUM_CAP)
            median = min(99.99, max(0.01, median))
            slope_var = 0.0
            if sxx > 1e-12 and n > 2 and resid > 0.0:
                slope_var = (resid / sxx) * (x_target - xbar) ** 2
            spread = math.sqrt(
                (SIGMA_RW ** 2) * horizon_days + resid + slope_var)
            spread = min(SPREAD_CAP, max(SPREAD_FLOOR, spread))
        # A saturated trophy (100% in the last read) stays saturated: the
        # launch-night refresh left it at 100%, so every newcomer holds
        # it too -- it is the entry trophy, not a rate that can dilute.
        # Its band is therefore tight (0.05 logit), not the wide
        # refresh/trend spread: a wide band around 99.99 pays pinball
        # loss at every quantile for a rate that cannot move.
        if saturated:
            spread = 0.05
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
