# HNAffect

Analysis code and aggregate data for the paper

> Keito Inoshita. *Emotional Change in AI-Related Online Discussions After ChatGPT Is
> Directionally Stable but Not Robust to Analytic Choices.*

The study asks how much of a reported before--after change in social media emotion is a
property of the community and how much is a property of the analysis. Starting from a
single interrupted time series fitted to AI-related Hacker News discussions, one source of
variation is added at a time — a control series of non-AI discussions, three emotion
instruments, repeated sampling, placebo calibration, and alternative specifications and
control groups — and the step at which the conclusion changes is reported.

## Repository layout

The code is released as it was executed, so the directory names of the original project
are kept, including the Japanese ones. Every script computes the project root as
`Path(__file__).resolve().parents[2]` and reads and writes relative to it. No executable
line was changed for this release; only a few comments that referred to the internal
workflow of the project were rewritten.

```
02_実験/code/      analysis scripts, numbered in the order they are run
02_実験/results/   aggregate outputs that reproduce every number in the paper
```

## What is included, and what is not

Included are the scripts and the aggregate outputs: monthly emotion means per group,
segmented-regression estimates for each of the ten samplings, bootstrap replicates,
placebo estimates, pseudo-treated topics, simulation results, the instrument-validation
tables, and `numbers.json`, from which the numbers in the manuscript are generated.

Not included are the raw Hacker News comments and the per-comment classifier scores.
These are several hundred megabytes and consist of text written by other people, so they
are not redistributed here. They are reproducible from the public Hacker News search API
with `01_collect_stories.py`, `02_sample_and_fetch.py` and `03_classify.py`; the sampling
rules and random seeds (2026 to 2035) are fixed in the scripts. The reference-annotator
files under `results/llm_annot/` identify comments by an item id only, without their text.

## Pipeline

| Script | Purpose |
| --- | --- |
| `01_collect_stories.py` | Collect Hacker News stories from the public search API |
| `02_sample_and_fetch.py` | Sample stories and comments per month and group |
| `03_classify.py` | Score comments with the three emotion instruments |
| `04_its.py` | Fit the single-series and comparative interrupted time series |
| `05_make_numbers.py` | Write `numbers.tex`, the single source of every number in the paper |
| `06_figures.py` | Draw the figures |
| `10_pooled.py`–`17_bootstrap.py` | Pooling, placebo, specification, topic controls, simulation, matched control, bootstrap |
| `18_title_check_stratified.py`–`22_mention_robustness.py` | Reference annotation, instrument validation, target and mention subsets |
| `23_supplement_tables.py`, `24_main_tables.py` | Generate the LaTeX tables |

Steps that only read `02_実験/results` — `24_main_tables.py` and `23_supplement_tables.py`
— run against this repository as it is. Running

```
python 02_実験/code/24_main_tables.py
```

on a fresh clone regenerates the four result tables of the paper, and they are
byte-identical to the ones in the submitted manuscript. The earlier steps need the raw
data to be collected first.

## Reference annotators

Human annotation was not available, so two locally run open-weight models,
Qwen2.5-14B-Instruct and Llama-3.1-8B-Instruct, were used as reference annotators for the
instruments, the emotion targets and the topic grouping. Their judgments are used to bound
measurement error, not as ground truth. No hosted API was used at any point.

## Disclosure

Claude Opus 5 (Anthropic) was used to write this analysis code, to draft the manuscript
and to create the figures. The author reviewed and edited all of it and takes full
responsibility for it. The same disclosure appears in the manuscript.

## License

MIT, see `LICENSE`.
