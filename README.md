# gta6-twitch-forecast-01-entry

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Iruwae/gta6-twitch-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Iruwae/gta6-twitch-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts Grand Theft Auto VI's launch-day (2026-11-19 UTC) Twitch concurrent viewers, hour by hour (hours 2–24), as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) per hour, from the first hour's count and the shapes of past big game launches. Entry for [make.fun/bounty/gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=817k0hr2&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

## How it works

`main.py` (Python 3.13, standard library only) builds its shape library from
the comparables already in each case file — 37 past big game launches' hourly
Twitch curves — and, per forecast hour, fits

```
log(viewers_h) = a_h + b_h * log(hour-1 print) + noise
```

over the launches of the same unlock regime (early-unlock when hour 1 exceeds
15,000 viewers, late-unlock otherwise). It is a hybrid, switched on the size
of hour 1: below 200,000 viewers it fits the line for the median by
least-absolute-deviation regression, with the residual standard deviation
as the spread (`exp(mu + z_tau * sd)`); above 200,000 it fits a separate
quantile regression at each of the five scored levels, because at the very
top end the quantiles' sensitivity to hour 1 differs by level. Slopes are
capped at 1, and console-only launches count at full weight while launches
with a PC version count at half weight (GTA VI is console-only).

Why this shape: launch-day viewership dips in the small hours, ramps at the
US unlock, then climbs to an evening peak. Hour 1's size reveals whether the
game had unlocked by 00:00 UTC — pre-unlock trickles grow 10–400x by evening,
midnight-UK/EU launches only 1.5–3x — so conditioning on it is what makes
the forecast beat persistence.

## Validation

Leave-one-out over the 37 comparables + the 4 practice cases, scored with
the contest's pinball loss on the log scale (lower is better):

| set | this model | house baseline |
| --- | --- | --- |
| practice cases p00–p03 | **0.216** | 0.387 |
| early-unlock LOO (27) | **0.244** | 0.321 |
| all-launch LOO (37) | **0.373** | 0.930 |

On the six LOO launches with hour 1 above 300,000 — the ones shaped like
the final — this model scores 0.119, against 0.145 for plain OLS and 0.139
for the median-only fit.

The baseline's practice loss of 0.387 here reproduces the 0.387 the rules
quote, so the scores are on the contest's scale. The program is
deterministic and runs in ~0.15 s per case (limit 8 s).
