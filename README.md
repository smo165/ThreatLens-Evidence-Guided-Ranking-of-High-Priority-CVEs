# ThreatLens-Evidence-Guided-Prioritization-and-Early-Warning-for-Exploitation-Relevant-CVEs
This is the artifact repo for the applied research paper of the same name
# ThreatLens Reproducibility Artifact

This repository contains the code and processed data needed to reproduce the results in:

**ThreatLens: Evidence-Guided Prioritization and Early Warning for Exploitation-Relevant CVEs**

The processed snapshot data included in this artifact are the canonical inputs for reproducing the paper results. The raw-data reconstruction scripts are also included for transparency, but public data sources may change over time, so rebuilding from raw sources may not exactly match the released processed data.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A GPU is recommended for reproducing neural ranker experiments. CPU is sufficient for rule baselines, learned non-neural baselines, structured baselines, and many analysis scripts.

---

## Main files

```text
Data/
  build_events_kev.py
  build_full_nvd_corpus.py
  build_snapshots_weekly_kev_forecast.py
  enrich_snapshots_with_epss.py
  enrich_snapshots_with_temporal_advisories.py
  enrich_snapshots_with_public_exploits.py
  ablate_nvd_text.py
  ablations/make_no_epss_ablation.py

experiments/
  evaluate_rule_baselines_samecutoff.py
  evaluate_learned_baselines_samecutoff.py
  evaluate_neural_prioritizer_transformer_samecutoff.py
  evaluate_early_warning_baselines_all.py
  analyze_early_warning_version_b.py
  analyze_early_warning_kfold.py
  evaluate_structured_logreg_ablations.py

data/
  processed/
  results/
```

The `data/processed/` directory contains the processed snapshot data used for the paper experiments.

---

## Reproducing results from processed data

Use the processed snapshot CSV and KEV event CSV included in `data/processed/`.

The main processed snapshot CSV should contain cutoff-aligned CVE-time snapshots. At minimum, the evaluation scripts expect columns such as:

```text
cve
cutoff_date
text
```

Depending on the experiment, the processed snapshot file may also contain enriched structured columns such as:

```text
epss_score
epss_percentile
advisory_count
public_exploit_available_by_cutoff
public_exploit_count
cvss_base_score
```

---

## Main ranking results

Run the rule baselines:

```bash
python experiments/evaluate_rule_baselines_samecutoff.py \
  --snapshots <path-to-final-processed-snapshots.csv> \
  --events <path-to-events-kev.csv> \
  --window_days 30 \
  --out_json <path-to-rule-baselines-output.json>
```

Run the learned non-neural baselines:

```bash
python experiments/evaluate_learned_baselines_samecutoff.py \
  --snapshots <path-to-final-processed-snapshots.csv> \
  --events <path-to-events-kev.csv> \
  --window_days 30 \
  --baselines structured_hgb,structured_logreg,tfidf_logreg \
  --out_json <path-to-learned-baselines-output.json> \
  --out_scores <path-to-learned-baselines-scores.csv>
```

Run the neural text ranker:

```bash
python experiments/evaluate_neural_prioritizer_transformer_samecutoff.py \
  --snapshots <path-to-final-processed-snapshots.csv> \
  --events <path-to-events-kev.csv> \
  --window_days 30 \
  --model_name BAAI/bge-large-en-v1.5 \
  --pooling cls \
  --hidden_dim 256 \
  --dropout 0.1 \
  --epochs 10 \
  --encoder_lr 1e-5 \
  --head_lr 1e-4 \
  --weight_decay 1e-4 \
  --pairs 50000 \
  --batch_size 64 \
  --eval_batch_size 256 \
  --max_length 512 \
  --seed 7
```

These scripts reproduce the same-cutoff ranking comparisons reported in the paper, including rule baselines, learned structured baselines, TF-IDF, and the neural text ranker.

---

## Early-warning results

Run non-neural forward-time and 10-fold early-warning baselines:

```bash
python experiments/evaluate_early_warning_baselines_all.py \
  --snapshots <path-to-final-processed-snapshots.csv> \
  --events <path-to-events-kev.csv> \
  --window_days 30 \
  --baselines random,cvss,public_exploit_count,epss,structured_combo,tfidf_logreg,structured_hgb,structured_logreg \
  --run_forward \
  --run_kfold \
  --folds 10 \
  --fold_universe target_event_cves \
  --score_scope relevant_cutoffs \
  --k_values 10,20,50,100 \
  --seed 7 \
  --out_dir <path-to-early-warning-baseline-output-dir>
```

This script evaluates the non-neural early-warning rows reported in the paper, including random, CVSS, public exploit count, EPSS, structured rule, TF-IDF, structured HGB, and structured logistic. It can run both the strict forward-time setting and the 10-fold KEV-target setting. Neural early-warning results are reproduced separately with the neural scripts below.

Run strict forward-time neural early warning:

```bash
python experiments/analyze_early_warning_version_b.py \
  --snapshots <path-to-final-processed-snapshots.csv> \
  --events <path-to-events-kev.csv> \
  --window_days 30 \
  --model_name BAAI/bge-large-en-v1.5 \
  --pooling cls \
  --hidden_dim 256 \
  --dropout 0.1 \
  --epochs 10 \
  --encoder_lr 1e-5 \
  --head_lr 1e-4 \
  --weight_decay 1e-4 \
  --pairs 50000 \
  --batch_size 64 \
  --eval_batch_size 256 \
  --max_length 512 \
  --seed 7
```

Run 10-fold neural early warning:

```bash
python experiments/analyze_early_warning_kfold.py \
  --snapshots <path-to-final-processed-snapshots.csv> \
  --events <path-to-events-kev.csv> \
  --window_days 30 \
  --folds 10 \
  --fold_universe target_event_cves \
  --model_name BAAI/bge-large-en-v1.5 \
  --pooling cls \
  --hidden_dim 256 \
  --dropout 0.1 \
  --epochs 10 \
  --encoder_lr 1e-5 \
  --head_lr 1e-4 \
  --weight_decay 1e-4 \
  --pairs 50000 \
  --batch_size 64 \
  --eval_batch_size 256 \
  --max_length 512 \
  --seed 7 \
  --k_values 10,20,50,100 \
  --score_scope relevant_cutoffs \
  --out_csv <path-to-per-cve-output.csv> \
  --out_json <path-to-summary-output.json> \
  --out_fold_csv <path-to-per-fold-output.csv>
```

Early-warning hits are counted only before KEV entry. The 30-day window is the training label horizon; the early-warning lead-time analysis itself is not capped at 30 days.

---

## Ablations

Create the No-EPSS text ablation:

```bash
python Data/ablations/make_no_epss_ablation.py \
  --input <path-to-final-processed-snapshots.csv> \
  --output <path-to-no-epss-snapshots.csv> \
  --verify
```

This removes the `TEMPORAL_EPSS_SIGNAL` block from the model-facing text column only. It does not change rows, labels, cutoffs, CVE IDs, or enrichment columns.

Create the NVD CVSS/CWE/CPE text ablation:

```bash
python Data/ablate_nvd_text.py \
  --input <path-to-final-processed-snapshots.csv> \
  --output <path-to-no-nvd-cvss-cwe-cpe-snapshots.csv> \
  --drop-cvss \
  --drop-cwe \
  --drop-cpe
```

The ablated snapshot files can then be passed to the same neural evaluation scripts used for the full neural text ranker.

Run structured logistic ablations:

```bash
python experiments/evaluate_structured_logreg_ablations.py \
  --snapshots <path-to-final-processed-snapshots.csv> \
  --events <path-to-events-kev.csv> \
  --window_days 30 \
  --variants full,no_epss,no_cvss \
  --out_dir <path-to-structured-logreg-ablation-output-dir>
```

Ablation experiments should use the same rows, labels, cutoffs, temporal/CVE-disjoint split protocol, model architecture, and training configuration as the corresponding full model, differing only in the targeted removed evidence.

---

## Rebuilding processed snapshots from raw sources

This step is optional for reproducing the paper tables. The released processed snapshots should be used for exact result reproduction.

To rebuild the data from raw public sources, obtain the following inputs.

---

## NVD

Download the NVD CVE 2.0 yearly JSON feed zip files, for example files named like:

```text
nvdcve-2.0-2002.json.zip
nvdcve-2.0-2003.json.zip
...
nvdcve-2.0-2026.json.zip
```

Then build the combined processed NVD corpus:

```bash
python Data/build_full_nvd_corpus.py \
  --input_dir <path-to-directory-containing-nvdcve-2.0-json-zips> \
  --output_json <path-to-combined-nvd-json> \
  --output_csv <path-to-combined-nvd-summary-csv>
```

The resulting combined JSON is passed to the weekly snapshot builder with:

```text
--nvd_json <path-to-combined-nvd-json>
```

---

## CISA KEV

Build the KEV event file:

```bash
python Data/build_events_kev.py \
  --output <path-to-events-kev.csv>
```

---

## EPSS

No manual EPSS download is required. The EPSS script queries the FIRST EPSS API and caches responses locally.

---

## GitHub Advisory Database

Obtain:

```text
github/advisory-database
```

Pass the internal advisory directory:

```text
--advisory_dirs <path-to-github-advisory-database>/advisories
```

The advisory script reads OSV-format JSON advisory records. If only the GitHub Advisory Database is supplied, the advisory source is GitHub Advisory Database advisories represented in OSV format.

---

## Exploit-DB

Obtain:

```text
https://gitlab.com/exploit-database/exploitdb.git
```

Pass:

```text
--exploitdb_csv <path-to-exploitdb>/files_exploits.csv
```

---

## Metasploit Framework

Obtain:

```text
https://github.com/rapid7/metasploit-framework.git
```

Pass:

```text
--metasploit_dir <path-to-metasploit-framework>/modules
```

---

## Full reconstruction command sequence

```bash
python Data/build_events_kev.py \
  --output <path-to-events-kev.csv>

python Data/build_full_nvd_corpus.py \
  --input_dir <path-to-directory-containing-nvdcve-2.0-json-zips> \
  --output_json <path-to-combined-nvd-json> \
  --output_csv <path-to-combined-nvd-summary-csv>

python Data/build_snapshots_weekly_kev_forecast.py \
  --events <path-to-events-kev.csv> \
  --nvd_json <path-to-combined-nvd-json> \
  --horizon_days 30 \
  --negatives_per_positive 50.0 \
  --output <path-to-weekly-snapshots.csv>

python Data/enrich_snapshots_with_epss.py \
  --snapshots <path-to-weekly-snapshots.csv> \
  --output <path-to-snapshots-with-epss.csv> \
  --cache_dir <path-to-epss-cache-dir>

python Data/enrich_snapshots_with_temporal_advisories.py \
  --snapshots <path-to-snapshots-with-epss.csv> \
  --output <path-to-snapshots-with-advisories.csv> \
  --advisory_dirs <path-to-github-advisory-database>/advisories

python Data/enrich_snapshots_with_public_exploits.py \
  --snapshots <path-to-snapshots-with-advisories.csv> \
  --output <path-to-final-enriched-snapshots.csv> \
  --exploitdb_csv <path-to-exploitdb>/files_exploits.csv \
  --metasploit_dir <path-to-metasploit-framework>/modules
```

---

## Notes

The processed data included with this artifact are the recommended inputs for reproducing the paper results.

The raw reconstruction scripts are provided to show how the processed data were created. Rebuilding from newly downloaded public sources may produce slightly different data because NVD, advisory databases, EPSS, Exploit-DB, and Metasploit can change over time.

The pipeline does not reconstruct exact historical NVD record revisions at each cutoff. Instead, NVD-derived fields are leakage-mitigated through text restrictions, removal of mutable or leakage-prone fields, and targeted ablations.

The public exploit enrichment uses Exploit-DB and Metasploit only to derive metadata features. Exploit code is not included in the model-facing text.
