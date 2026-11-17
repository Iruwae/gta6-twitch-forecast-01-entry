# GTA VI Twitch CCU forecaster — make.fun entry

Forecasts Grand Theft Auto VI's launch-day (2026-11-19 UTC) Twitch concurrent
viewers, hour by hour (hours 2–24), as five quantiles (0.05 / 0.25 / 0.5 /
0.75 / 0.95) per hour. Entry for
[make.fun/bounty/gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).

## How it works

`main.py` (Python 3.13, standard library only) builds its shape library from
the comparables already in each case file — 37 past big game launches' hourly
Twitch curves — and, per forecast hour, fits

```
log(viewers_h) = a_h + b_h * log(hour-1 print) + noise
```

over the launches of the same unlock regime (early-unlock when hour 1 exceeds
15,000 viewers, late-unlock otherwise), with the residual standard deviation
as the spread. Quantiles are `exp(mu + z_tau * sd)`.

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
| practice cases p00–p03 | **0.227** | 0.387 |
| early-unlock LOO (27) | **0.252** | 0.321 |
| all-launch LOO (37) | **0.383** | 0.930 |

The baseline's practice loss of 0.387 here reproduces the 0.387 the rules
quote, so the scores are on the contest's scale. The program is
deterministic and runs in ~0.15 s per case (limit 8 s).

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
