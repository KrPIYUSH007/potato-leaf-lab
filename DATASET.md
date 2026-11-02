# Dataset preparation — Phase II

## Source and attribution

**Potato Leaf Disease Dataset in Uncontrolled Environment**, Mendeley Data,
version 1 (10 November 2023), DOI [10.17632/ptz377bwb8.1](https://doi.org/10.17632/ptz377bwb8.1).
Contributors: Nabila Husna Shabrina, Siwi Indarti, Rina Maharani, Dinar Ajeng
Kristiyanti, Irmawati Irmawati, Niki Prastomo, and Tika Adillah M.

The repository lists 3,076 smartphone images from Central Java, Indonesia,
with seven broad categories. Licence: **CC BY 4.0**. Preserve this attribution
when using the dataset and identify changes to its labels, images or selection.
Dataset licensing applies to the images; it is not a licence for this project's code.

Primary paper: [Sujatha et al., Scientific Reports (2025)](https://www.nature.com/articles/s41598-024-72197-2).

## Reproduce the preparation

Use Python 3.12. The tested environment is recorded in `requirements-data-lock.txt`.
Run from the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements-data-lock.txt
& ./.venv/Scripts/python.exe -m leaflab download --workers 8
& ./.venv/Scripts/python.exe -m leaflab prepare --out reports/data-v1
& ./.venv/Scripts/python.exe -m leaflab validate --manifest reports/data-v1/manifest.csv --data data/raw
& ./.venv/Scripts/python.exe -m pytest -q
```

For an existing published manifest, download the dataset and run `validate`; do
not regenerate partitions merely to get a different model score. `prepare`
refuses to overwrite an existing output directory. For a deliberate revised
audit, use a new versioned directory and explain the change.

On Linux/macOS use `.venv/bin/python` in place of the Windows executable.
The standard-library downloader uses the dataset's public file catalogue,
checks each file's SHA-256 against publisher metadata, retries transient failures,
and reuses only previously verified files. The public endpoint is not a permanent
API guarantee; changes should produce errors rather than silently substitute data.

## Labels and file layout

```text
data/raw/
  bacteria/
  fungi/
  healthy/
  nematode/
  pest/
  phytophthora/
  virus/
  source_catalogue.json
reports/data-v1/
  manifest.csv
  audit.json
  classes.json
  near_duplicate_candidates.csv
```

The source folder spells `Phytopthora`; our canonical label corrects it to
`phytophthora`. This spelling correction does not change category membership.
All manifest paths are relative to `data/raw`, using forward slashes.
Class IDs are their indices in `classes.json` and must be saved in model checkpoints.

## Audit and split rules

1. Decode images with Pillow, apply EXIF orientation, convert to RGB, record
   size, file SHA-256, decoded-pixel SHA-256 and 64-bit difference hash (dHash).
2. Exclude unreadable images. Quarantine every copy of identical oriented RGB
   pixels assigned to conflicting categories, without selecting a label. Record
   all exclusions for human review; leave source files untouched. Deduplicate
   remaining exactly equal pixels within a category.
3. Connect candidate near-duplicate pairs whose dHash Hamming distance is at
   most four, including transitive connections. Keep connected groups together
   in a partition; retain non-identical images. This conservative rule may group
   visually different images, so inspect the review CSV/contact sheets.
4. Optionally add known same-leaf/plant/session groups using `--groups groups.csv`,
   where the CSV contains `path,group`. These are merged with automatic groups.
5. With seed 42, use group-disjoint candidate splits targeting 70/15/15. Choose
   balance by class counts only, without any model scores. Require every class
   in each partition. Group integrity takes priority over exact percentages.
6. Freeze the CSV and its SHA-256. Calculate class weights from training counts
   only using N/(K*n_c). All later models use this same manifest.

The validator checks the manifest against its frozen audit hash and rejects unsafe
paths, duplicate paths, cross-partition groups, exact image leakage and missing
categories. With `--data`, it rechecks every
file's hash. Training-only augmentation will be added in the model phase.

## Outputs and limits

`audit.json` is the machine-readable result. `category_samples.jpg` and bounded
`near_duplicates_*.jpg` sheets are local inspection material, not model results.
Raw images, source download URLs, checkpoints and local inspection images are
excluded from Git. The public-facing audit uses relative image paths, not local
machine paths.

dHash is a screening heuristic, not a complete duplicate or same-plant detector.
The source file timestamps do not establish leaf/plant identity. Without verified
plant/session metadata, these partitions cannot establish plant-independent
generalisation. Review candidates and any available provenance before making
field-performance claims. Do not repeatedly inspect test images to tune models.

This split differs from the paper's stratified 10-fold evaluation. Phase II
produces data and integrity evidence only; it does not train a model or measure
classification accuracy.

Implementation references:
[Scikit-learn group splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html),
[Pillow EXIF orientation](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.exif_transpose).
