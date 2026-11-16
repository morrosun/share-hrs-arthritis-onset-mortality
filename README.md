# Age at first reported diagnosis of arthritis and mortality in two ageing cohorts

Analysis code for a study of whether the reported age at first diagnosis of arthritis
predicts mortality in later life, and of how much the answer depends on the analysis
rather than on the disease.

The study uses two population-based cohorts of ageing, SHARE (Survey of Health, Ageing
and Retirement in Europe) and HRS (Health and Retirement Study), and treats the
time scale, left truncation, cohort design and shared age support as part of the
estimand rather than as technical choices.

## What the code does

* Builds the analytic cohorts from the harmonised SHARE and HRS files, including the
  handling of same-year deaths, the 2020 SHARE entry wave, and the onset interval in HRS.
* Fits Cox models with age as the time scale and delayed entry, and the same contrasts
  with follow-up time as the scale for comparison.
* Re-estimates the onset contrasts under three follow-up rules and within shared age
  windows, and reports a contrast as not estimable when the groups do not overlap.
* Quantifies how many unrecorded deaths would move the main estimate to the null.
* Produces the figures and the numbers reported in the manuscript and its
  supplementary tables.

Principal results, for orientation: the same SHARE cohort yields a hazard ratio of 4.97
(4.52-5.46) for onset at 75 years or later when follow-up time is used without left
truncation, and 0.82 (0.74-0.91) when age is the time scale; any arthritis in HRS
carries 1.33 (1.28-1.39); the shared-window HRS contrasts are 1.04 and 0.98 under the
primary rule and 0.609 and 0.611 when a second interview is required.

## Repository layout

```
analysis/          Python and R scripts, numbered in the order they were run
README.md
LICENSE
.zenodo.json
```

Scripts are numbered `s1` to `s107` and form a pipeline: several middle steps
read CSV files written by earlier steps. `analysis/README.md` lists the stages,
the dependencies, the cohort definitions and the follow-up rules. The scripts a
reader is most likely to want:

| Script | Purpose |
|---|---|
| `s86_share_final.py` | SHARE prevalent cohort under the final rules; the 0.82 estimate |
| `s85_hrs_final.py` | HRS mortality models by arthritis state; the 1.33 estimate |
| `s94_hrs_ruleB.py` | HRS shared-window contrasts under the follow-up rules |
| `s99_ruleB_ladder.py` | nine-specification ladder for the two HRS window estimates |
| `s104_boundary_sensitivity.py` | deaths at the lower boundary: dropped, retained, entry shifted |
| `s107_boundary_mechanism.py` | why the censored variant reproduces the reported values |
| `s96_window_audit.py`, `s103_window_ci.py` | shared-window audit and CI decomposition |
| `s69_followup_rules.py` | follow-up rules A, B and C |
| `fig1_v18.R`, `fig2_v18.R`, `fig3_v18.R`, `fig4.R` | figures |

## Data

The code reads harmonised cohort files named `share.RData`, `hrs.RData`, `elsa.RData`
and `charls.RData` from the directory given by `AGING_DB_DATA`. These files are not
distributed here. They are available to registered users from the Gateway to Global
Aging Data (https://g2aging.org) after the data-use agreements of the individual
studies.

**No participant-level data are included in this repository**, and none are needed to
read the code. The repository also does not contain the manuscript builders, the table
rendering code, or any intermediate file derived from participant records.

## Requirements

* Python 3.11 with `pandas`, `numpy`, `scipy`, `lifelines` (0.30), `statsmodels` (0.14),
  `pyreadr`; the versions used for the manuscript are in `requirements.txt`
* R 4.6 with `ggplot2`, `dplyr`, `scales`
* PostgreSQL client only for `s5_charls_raw_probe.py`, which reads a locally loaded
  CHARLS table; it is not needed for the main analysis

## Running

Set two environment variables, then run the scripts in numeric order:

```bash
export AGING_DB_DATA=/path/to/harmonised/RData     # input: share.RData, hrs.RData, ...
export AGING_DB_OUT=/path/to/output                # defaults to ./output
python analysis/s86_share_final.py
```

R scripts use `AGING_DB_ROOT` as their working directory:

```bash
export AGING_DB_ROOT=/path/to/repository
Rscript analysis/fig3_v18.R
```

Each Python script prints a plain-text report and writes the same text, plus any
CSV it produces, to `AGING_DB_OUT` (created if absent). The R scripts then read
the Python CSV output from `$AGING_DB_ROOT/out` and write the figures there.
Estimates are stored at three decimals and displayed in the manuscript at two;
`analysis/README.md` gives the headline values each script should reproduce.

## What is deliberately not included

* Participant-level or respondent-level data of any kind.
* Manuscript and supplementary-material builders, HTML/PDF rendering, and the
  table-formatting layer. These contain unpublished text and are not analysis code.
* Local configuration, credentials and session artefacts. Database access is read from
  environment variables.

## Citation

If you use this code, please cite the article (citation and DOI will be added when the
article is published) and this repository:

> Bi C, Zhong Y, Li J, Meng D, Wang K. Analysis code for: Age at first reported
> diagnosis of arthritis and mortality in two ageing cohorts. Zenodo.
> https://doi.org/10.5281/zenodo.24031958

The concept DOI 10.5281/zenodo.24031958 always resolves to the latest release.
Version 1.1.1 is the release used for the manuscript; the v1.0.0 snapshot remains
at 10.5281/zenodo.24031959. Archived snapshots are frozen at their tags; this
README is a living document and may be ahead of them.

## License

MIT. See `LICENSE`.
