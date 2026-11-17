"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only).

Model: comparable-launch shape regression.

Launch-day Twitch viewership does not stay flat after hour 1: it dips in the
small hours, ramps when the US unlocks (hours 5-8), dips again around midday,
then climbs to an evening peak (hours 19-21) before fading. The shape depends
strongly on how big hour 1 already is, because hour 1's size tells you
whether the game had already unlocked by 00:00 UTC: launches whose hour 1 is
a pre-unlock trickle (a few hundred viewers) grow 10-400x by evening, while
launches that unlocked at midnight in the UK/Europe (tens or hundreds of
thousands in hour 1) grow only 1.5-3x. Across the 37 past launches in the
case's comparables, corr(log hour-1, log-ratio to hour h) runs from -0.1 at
hour 2 to -0.94 at hour 20: the absolute level of a later hour is almost
unrelated to hour 1 across all launches, but strongly related once you
condition on the unlock regime.

So, per forecast hour h = 2..24, this program fits an ordinary least squares
line over the comparable launches of the same regime:

    log(viewers_h) = a_h + b_h * log(p1) + noise

where p1 is the case's hour-1 print, and the spread is the residual
standard deviation of that fit. The five quantiles are

    q_tau(h) = exp(mu_h + z_tau * sd_h)

with z_tau the standard normal quantiles at 0.05/0.25/0.5/0.75/0.95.

Regime: a launch counts as early-unlock when its hour-1 print exceeds
15,000 viewers (the comparables split cleanly there: late-unlock launches
top out at ~10,600 in hour 1, early-unlock ones start at ~20,500). The
case's regime is decided by its own hour-1 print, and the fit uses only
launches of that regime; if a regime has fewer than 5 usable launches, all
launches are used. A comparable contributes to hour h only when it has a
valid print for that hour, so sparse categories simply vote on fewer hours.
The Grand Theft Auto VI category's own pre-release history (one of the
comparables) has no release-day hour 1 and is excluded automatically.

Validation (leave-one-out over the 37 comparables + the 4 practice cases,
scored with the contest's pinball loss on the log scale; lower is better):

    practice cases (p00-p03):  this model 0.227, house baseline 0.387
    early-unlock LOO (27):     this model 0.252, house baseline 0.321
    all-launch LOO (37):       this model 0.383, house baseline 0.930

The house baseline's practice loss of 0.387 here reproduces the 0.387 the
rules quote, so the scoring above is on the same scale as the contest.

The program reads every <id>.in in the input directory and writes one
<id>.out per case, exactly as the starter does. It is deterministic (no
randomness, sorted iteration) and runs in milliseconds: the only work is a
few dozen small regressions over the comparables already in the case file,
so there is nothing to vendor and no network is needed.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Hour-1 print above which a launch counts as already unlocked at 00:00 UTC.
EARLY_THRESHOLD = 15000
# A regime with fewer usable launches than this falls back to all launches.
MIN_REGIME_SIZE = 5


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def release_day_hours(comparable: dict) -> dict[int, float]:
    """Hour -> viewers for hours 1..24 of the comparable's release day (UTC).

    A point covers the hour starting at its `from`. Hours with no point, or
    with a non-positive viewer count, are left out.
    """
    day_start = parse_time(comparable["day_start"])
    hours: dict[int, float] = {}
    for point in comparable.get("hourly") or []:
        viewers = point.get("viewers")
        if not viewers or viewers <= 0:
            continue
        offset = (parse_time(point["from"]) - day_start).total_seconds() / 3600
        if 0 <= offset < 24:
            hours[int(offset) + 1] = float(viewers)
    return hours


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares intercept and slope of y on x (slope 0 when x is constant)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    sxx = sum((x - mean_x) ** 2 for x in xs)
    slope = sxy / sxx if sxx > 0 else 0.0
    return mean_y - slope * mean_x, slope


def pstdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.5
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def persistence_quantiles(p1: float, hour: int) -> list[float | int]:
    """The house baseline's forecast: hour 1's print with a widening spread."""
    return [round2(max(1.0, p1) * math.exp(z * 0.15 * math.sqrt(hour - 1))) for z in Z]


def hour_quantiles(library: list[dict], p1: float, hour: int) -> list[float | int]:
    """The five quantiles of hour `hour`'s viewers, given hour-1 print p1."""
    points = [
        (math.log(launch["p1"]), math.log(launch["hours"][hour]))
        for launch in library
        if hour in launch["hours"]
    ]
    if not points:
        # No comparable has this hour: fall back to the persistence spread.
        return persistence_quantiles(p1, hour)
    intercept, slope = ols([p[0] for p in points], [p[1] for p in points])
    mu = intercept + slope * math.log(max(1.0, p1))
    residuals = [y - (intercept + slope * x) for x, y in points]
    sd = pstdev(residuals)
    return [round2(min(1e9, math.exp(mu + z * sd))) for z in Z]


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is not None and first.get("viewers"):
        p1 = max(1.0, float(first["viewers"]))
    else:
        # No hour-1 print: use the last whole-hour point before the
        # checkpoint, else 1. The case is unscorable anyway, but the output
        # must still be valid — an invalid case takes no prize.
        hourly = case.get("history_hourly") or []
        p1 = max(1.0, float(hourly[-1]["viewers"])) if hourly else 1.0

    # The shape library: every comparable with a valid release-day hour 1.
    library = []
    for comparable in case.get("comparables") or []:
        hours = release_day_hours(comparable)
        if 1 in hours:
            library.append({"p1": hours[1], "hours": hours})
    if len(library) < MIN_REGIME_SIZE:
        # Too little shape data to fit on: emit the valid persistence
        # forecast instead of failing the case outright.
        return {
            "hours": [
                {"hour": row["hour"], "q": persistence_quantiles(p1, row["hour"])}
                for row in case["forecast"]
            ]
        }

    # Regime: early-unlock launches are fitted separately from late ones.
    early = [launch for launch in library if launch["p1"] > EARLY_THRESHOLD]
    late = [launch for launch in library if launch["p1"] <= EARLY_THRESHOLD]
    regime = early if p1 > EARLY_THRESHOLD else late
    if len(regime) < MIN_REGIME_SIZE:
        regime = library

    return {
        "hours": [
            {"hour": row["hour"], "q": hour_quantiles(regime, p1, row["hour"])}
            for row in case["forecast"]
        ]
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
