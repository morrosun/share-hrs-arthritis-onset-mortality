# Analysis scripts

The scripts are numbered in the order they were run. They form a pipeline, not a
set of independent programs: several middle steps read CSV files written by
earlier steps. Run them in numeric order, or use the stage list below.

All Python scripts read the harmonised cohort files from `AGING_DB_DATA` and
write their text report and CSV output to `AGING_DB_OUT` (default: `./output`,
created if absent). The R scripts read the CSVs written by the Python steps from
`$AGING_DB_ROOT/out` and write the figures there.

```bash
export AGING_DB_DATA=/path/to/harmonised/RData
export AGING_DB_OUT=/path/to/output
export AGING_DB_ROOT=/path/to/repository
python analysis/s86_share_final.py
```

## What the headline numbers need

| Output | Script | Key printed values |
|---|---|---|
| SHARE cohort and the 0.82 estimate | `s86_share_final.py` | 18,587 respondents, 4,932 deaths, 131,654 person-years; 1.02 (0.96–1.09) and 0.82 (0.74–0.91) |
| HRS cohort and the 1.33 estimate | `s85_hrs_final.py` | 34,568 respondents, 13,389 deaths; any arthritis 1.333 (1.282–1.385); five-state 1.371 / 1.151 / 1.146 / 1.295 |
| SHARE follow-up rules A/B/C | `s69_followup_rules.py` | A 18,587 / 4,932; B 16,851; C 23,894 |
| HRS shared-window contrasts | `s94_hrs_ruleB.py` | 1.041 and 0.979 under rule A; 0.609 and 0.611 under rule B |
| Nine-specification ladder | `s99_ruleB_ladder.py` | 0.6086 and 0.6108 fully adjusted; difference 0.0022–0.0289 |
| Boundary deaths | `s104_boundary_sensitivity.py`, `s107_boundary_mechanism.py` | 0.6108 dropped, 0.6103 retained, 0.6108 censored |
| Tipping point | `s92_checks.py` | 115 of 325 censored records (35.4%) move the estimate to 1.001 |
| Figures 1–4 | `fig1_v18.R`, `fig2_v18.R`, `fig3_v18.R`, `fig4.R` | require the CSVs from s37/s93, s90 and s34 |

Estimates are stored in the CSV files at three decimals and displayed in the
manuscript at two decimals. For example, the HRS ≥75 incident estimate is
1.2946 unrounded, 1.295 at three decimals and 1.30 at two.

## Stages

| Steps | Purpose |
|---|---|
| `s0`–`s3` | Exploratory variable scan and feasibility checks. `s0` skips harmonised files that are absent; `s1` and `s2` also inspect MHAS and KLOSA if present. |
| `s4`–`s20` | First full analysis: time scale, left truncation, prevalent vs incident designs, follow-up rules, duration. |
| `s21`–`s39` | Shared-support audits and the v5 cohort definitions (entry at the first positive report, onset at the interval midpoint). |
| `s42`–`s52` | Fine-grained support, 75–79 restriction, wave sensitivity, duration estimand, multiplicity, risk-set and PH audits. |
| `s61`–`s78` | Sample reconciliation, selection/IPW, complete-case checks, administrative censoring, follow-up rules, same-year deaths. |
| `s85`–`s87` | Final SHARE and HRS cohorts under the rules used in the paper, plus two sensitivity rows. |
| `s90`–`s107` | Final figures, period-specific PH estimates, tipping point, window audit, rule-B verification, boundary-death sensitivity and mechanism. |

## Definitions used in the code

* **Entry and exit.** Entry is the later of age 40 and the first interview (SHARE)
  or the first positive arthritis report (HRS incident cohort). Exit is age at
  death, otherwise age at the last interview. Records with exit ≤ entry are
  removed; a death in the year of entry is retained with 0.5 years.
* **Onset.** SHARE: `radiagarthr`, restricted to ages ≥5. HRS incident cohort:
  midpoint of the interval between the last negative and first positive report.
  Onset groups are <60, 60–74 and ≥75 years.
* **SHARE exclusions.** No year of birth; entry before age 40; entry in the 2020
  wave (wave 8); exit after age 110.
* **HRS exclusions.** No valid arthritis report or age; onset-interval midpoint
  before age 40; missing education.
* **Rule A (main).** Censor at the last interview, use recorded deaths, retain
  same-year deaths with 0.5 years.
* **Rule B.** Require at least two interviews. This removes person-time together
  with deaths and is a selection diagnostic, not an estimate of the same quantity.
* **Rule C.** Assume that respondents without a death record survived to 2021.
  This adds person-time in which no death was recorded.
* **Shared window.** Five-year age bands enter a window only if both groups
  contribute at least 20 person-years; one-year bands and continuous attained age
  are examined as specifications. A contrast is not estimated when no band
  qualifies or when the reference group contributes fewer than five deaths.

## Dependencies and superseded scripts

* `s7_export.py` writes `analysis_share.csv`. `s10`, `s11`, `s15`, `s16`, `s17`,
  `s22`, `s23`, `s25` and `s26` read it; run `s7` first.
* `s92_checks.py` reads `tableS6_duration_v5.csv` from `s48_duration_estimand.py`.
* `s99_ruleB_ladder.py` reads the frame CSVs it writes itself and optionally calls
  `Rscript` for an independent Cox refit; the R step is skipped when `Rscript` is
  not on the path.
* `s22_final.py`, `s25_v2.py`, `s26_table2.py`, `s33_v5_table5.py` and
  `s34_v5_outputs.py` are v5-era working scripts. Their outputs are inputs to
  later steps, but the numbers reported in the paper come from `s85`, `s86`,
  `s94`, `s99`, `s104` and `s107`.

## Software

Python 3.11 with pandas, numpy, scipy, lifelines 0.30, statsmodels 0.14 and
pyreadr (see `requirements.txt`). R 4.6 with ggplot2, dplyr and scales for the
figure scripts. The PostgreSQL probe `s5_charls_raw_probe.py` is optional and
needs a local CHARLS table; it is not used in the main analysis.
