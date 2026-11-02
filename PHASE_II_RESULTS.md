# Phase II — Dataset preparation results

Prepared 2 November 2026. This phase prepares data; no classifier has been trained.

## Outcome

Downloaded and verified all **3,076 source images** from Mendeley Data version 1
(**754,363,044 bytes**). Every source file's SHA-256 matched publisher metadata.
All 3,076 images decoded successfully; no corrupt images were found.

After documented exclusions, **3,015 images** remain in the experiment manifest.
Source files are preserved locally. Raw images and inspection sheets are excluded
from GitHub; scripts, relative-path manifests and reports are committed.

## Data-quality findings

- **Six exact-image groups have conflicting category assignments (12 files).**
  Three occur under fungi/pest, one under healthy/pest, one under nematode/pest,
  and one under pest/virus. Every conflicting copy was quarantined from this
  experiment rather than selecting a label. These are inconsistencies for the
  proposed single-label task, not proof of which category is biologically correct.
- **49 redundant same-category images were excluded.** One deterministic copy
  of each exact oriented-RGB image remains.
- After excluding conflicting images, the audit identified **87 candidate pairs**
  at dHash distance at most four: 49 exact pairs and 38 non-identical pairs.
  Candidate connections were grouped transitively; related candidates stay in one
  partition. The retained manifest contains **2,983 groups**.
- No mixed-category near-duplicate candidate pairs remained after conflict
  quarantine. This does not establish that all labels are correct.
- Category sample sheets and candidate sheets 1 and 9 were visually inspected.
  They show variable backgrounds, hands, whole-plant views and near-identical
  captures. This is a limited visual inspection, not an expert label audit or an
  exhaustive same-plant investigation. Conservative grouping was retained.

The source folder `Phytopthora` was mapped to canonical `phytophthora`, with no
change to category membership. Exclusion paths and hashes are recorded in
[audit.json](reports/data-v1/audit.json).

## Frozen partitions

Seed: **42**. Group-disjoint candidate partitions were selected using category
balance only, without training a model. The approximate 70/15/15 target gives:

| Category | Source | Retained | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| Bacteria | 569 | 569 | 402 | 82 | 85 |
| Fungi | 748 | 740 | 520 | 111 | 109 |
| Healthy | 201 | 199 | 139 | 31 | 29 |
| Nematode | 68 | 67 | 46 | 9 | 12 |
| Pest | 611 | 601 | 419 | 87 | 95 |
| Phytophthora | 347 | 309 | 219 | 47 | 43 |
| Virus | 532 | 530 | 367 | 86 | 77 |
| **Total** | **3,076** | **3,015** | **2,112** | **453** | **450** |

Train/validation/test proportions are approximately **70.05% / 15.02% / 14.93%**.
Every category occurs in every partition. The nematode category remains small;
its test recall will be uncertain with just 12 examples.

The validator confirmed no path duplication, cross-partition exact image hashes
or cross-partition group overlap, and rechecked the retained image file hashes.
The manifest is protected against unnoticed edits by this SHA-256:

```text
9dbbe9a5bdcafce0cf9ce7d9ed8467b9fcecb5d3e79543e77c054314415ceb75
```

Use the same [manifest](reports/data-v1/manifest.csv) for every later model. Class
weights in the audit are calculated from **training counts only**. Do not augment
before splitting or use test results to choose model settings.

## Environment and validation

- Python **3.12.14**, with data dependencies pinned in
  [requirements-data-lock.txt](requirements-data-lock.txt).
- **Seven automated tests passed**, covering duplicate conflicts, quarantine,
  transitive grouping, reproducible/disjoint splits, corrupt images, changed
  images/manifests, unsafe paths and verified-download reuse.
- The actual 3,015-row manifest passed image-hash and partition validation.
- Laptop hardware reports an **NVIDIA GeForce RTX 3050 with 6,144 MiB GPU memory**
  and approximately **15.7 GiB RAM**. Driver version: **581.80**.
- PyTorch and CUDA execution have **not** been tested or installed in this data
  environment. GPU visibility in the driver does not establish framework support.

See [environment.json](reports/environment.json) and
[download_verification.json](reports/download_verification.json).

## Interpretation and remaining limits

These findings are reasons for careful preparation, not evidence that a CNN will
beat the paper. The cleaned image selection and held-out protocol differ from
the paper's 3,076-image, stratified 10-fold experiment. Compare our methods within
the same cleaned protocol and disclose the differences when citing 62.6%.

dHash does not identify every related image. Verified leaf/plant/session IDs are
not available in this preparation, and the manifest does not demonstrate
plant-independent or geographically independent generalisation. Team/guide review
of grouping assumptions and the quarantine decision remains appropriate before
final model experiments. A deliberately revised split must receive a new version;
do not silently change this one after looking at scores.

**Next phase:** verify a compatible PyTorch environment and GPU access, implement
frozen InceptionV3 feature extraction plus SVM on the shared manifest, and select
settings on validation data. Reserve the final test evaluation until settings are
locked. Custom CNN experiments and the Streamlit demonstration follow.

## Source and reproduction

[Mendeley dataset version 1](https://data.mendeley.com/datasets/ptz377bwb8/1),
DOI 10.17632/ptz377bwb8.1, CC BY 4.0. Full attribution and commands are in
[DATASET.md](DATASET.md). No model accuracy is reported in this phase.
