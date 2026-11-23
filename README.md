# Potato Leaf Lab

Custom CNN and InceptionV3 + SVM experiments for seven-category potato leaf classification.

## Current status---

Research, dataset preparation and the InceptionV3 + SVM baseline are complete.
The selected baseline achieved **68.87% validation accuracy** and **63.36% validation macro-F1** on 453 images. The 450-image test partition remains reserved for final evaluation. Custom CNN training is next.

- [Development tracker](DEVELOPMENT_TRACKER.md) covers milestones, completed work and pending tasks.
- [Baseline results](PHASE_III_RESULTS.md) records the actual validation metrics, environment and remaining work.
- [Baseline setup and commands](BASELINE.md) explains training, saved models and image prediction.
- [Latest Review 1 presentation](docs/presentations/Plant_Leaf_Review1_Literature_Updated.pptx).
- [Research review document](docs/research/Potato_Leaf_Project_Research_Review.docx).
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
an improvement over the publication. The baseline has been implemented and evaluated on validation data.
Custom CNN training and the Streamlit demonstration follow.
