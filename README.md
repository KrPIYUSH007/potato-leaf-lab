# Potato Leaf Lab

Custom CNN and InceptionV3 + SVM experiments for seven-category potato leaf classification.

## Current status

Phase I research plan and Phase II dataset preparation tools are available.
No classification model has been trained and no project accuracy has been measured.

- [Phase I research and scope](PHASE_I_RESEARCH_AND_SCOPE.md): paper, objectives, proposed responsibilities and guide decisions.
- [Extended literature review](LITERATURE_REVIEW.md): additional potato-specific journal papers, stronger same-dataset references and proposed team reading assignments.
- [Dataset setup and audit procedure](DATASET.md): source attribution, downloads, duplicate handling and reproducible partitions.
- [Phase II results](PHASE_II_RESULTS.md): actual audit findings, split counts and remaining limits.
- [Frozen image manifest](reports/data-v1/manifest.csv) and [machine-readable audit](reports/data-v1/audit.json).

## Quick start

Use Python 3.12 from the repository root:

```powershell
py -3.12 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements-data-lock.txt
& ./.venv/Scripts/python.exe -m leaflab download
& ./.venv/Scripts/python.exe -m leaflab validate --data data/raw
& ./.venv/Scripts/python.exe -m pytest -q
```

The published manifest is already prepared. To intentionally create a new audit,
run `python -m leaflab prepare --out reports/data-v2` and document why it changed.
Raw images remain local and are not included in Git. The download is about 754 MB.

## Planned model experiments

1. InceptionV3 features + SVM, based on the selected paper.
2. Custom CNN trained from scratch with ordinary loss.
3. The same CNN with class-weighted loss.

All use common partitions. The initial held-out evaluation and dataset exclusions
differ from the paper's protocol, so a score above its 62.6% does not alone establish
an improvement over the publication. The next phase implements the baseline;
CNN training and the Streamlit demonstration follow.
