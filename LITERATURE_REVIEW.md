# Extended literature review — potato leaf classification

Prepared **3 November 2026**. Five additional journal articles are discussed below,
along with the dataset-origin paper. These supplement the original primary paper;
no model implementation or project accuracy is claimed here.

## Main finding for our project

**62.6% is the result of our selected InceptionV3 + SVM reference, not the best
reported result on this dataset.** Chang and Lai (2024) report a 89.87% mean
cross-validation accuracy and a 90.68% highest test result using RegNetY-400MF on
the same 3,076-image, seven-category collection. Mondal et al. (2026) also study
that collection with a hybrid architecture. This broader literature must be
acknowledged when defining our research gap.

The numbers below use different protocols, model-selection procedures and image
selections. They are reported findings, not a fair numerical leaderboard, and do
not predict our results on the cleaned 3,015-image manifest.

## Reading list and comparison

| ID | Paper / journal / year | Data and method | Reported result | What we learn |
|---|---|---|---|---|
| P0, existing primary | Sujatha et al., *Scientific Reports*, 2025 | Several crop datasets; potato: InceptionV3 features + SVM | 62.6% potato accuracy; stratified 10-fold evaluation | Our paper-based baseline; not a state-of-the-art ceiling |
| P1 | Chang & Lai, *Machine Learning and Knowledge Extraction*, 2024 | Same seven-category Mendeley dataset; RegNetY-400MF, transfer learning, augmentation | 89.87% mean, SD 1.28 in cross-validation comparison; highest individual test result 90.68% | Essential lightweight, same-dataset reference |
| P2 | Jlassi et al., *SN Computer Science*, 2024 | PlantVillage; MobileNetV2, reweighting and explainability | Abstract reports 98.61% accuracy | Direct precedent for imbalance handling; different dataset |
| P3 | Mondal et al., *Scientific Reports*, 2026 | PlantVillage and the same seven-category Mendeley dataset; DenseNet169 + Transformer + adaptive activation | 99.54% and 87.50% respectively; statistical section reports five random-split experiments | Architecture/cost trade-off; controlled vs field-image performance |
| P4 | Jain et al., *Scientific Reports*, 2026 | Three potato classes from PlantVillage; CNN ensemble with optimizer diversity and augmentation | 97% for EDL10 | Training choices and ensemble trade-offs; examine augmentation/split independence |
| P5 | Talab et al., *Discover Applied Sciences*, 2024 | Separate 1,900-image, three-category collection; image features + Random Forest, resampling | Original 87.89%; oversampling 91.09%; undersampling 96.17% | Imbalance strategies can change results; verify comparability of evaluation sets |

P1–P5 are journal articles from MDPI or Springer Nature, within the guidelines'
2024–2026 window. They are not conference base papers. The dataset-origin article
P6 has a publication-date qualification explained below.

## P1 — Most important additional reading

**Chang, C.-Y., & Lai, C.-C. (2024). Potato Leaf Disease Detection Based on a
Lightweight Deep Learning Model.** *Machine Learning and Knowledge Extraction,
6*(4), 2321–2335. Published **14 October 2024**; 26 August is its submission date,
not publication date. DOI: [10.3390/make6040114](https://doi.org/10.3390/make6040114).
[Open publisher full text](https://www.mdpi.com/2504-4990/6/4/114).

**Plain explanation:** Reuse a relatively small pretrained neural network and
adapt it to the potato task, rather than train a large network from zero.

**What the authors did:** Studied RegNetY-400MF on the same seven-category
uncontrolled-environment dataset. They investigated transfer learning, batch size,
learning rate and training-image augmentation. Their selected batch size and
learning rate were 16 and 0.001; initial settings in Section 3.1 differ, so read
the tuning sections rather than copying only the initial configuration.

**Results to distinguish:** Table 5 lists ten test results, with 90.68% the highest
and 89.87% their average. Section 3.7 and Table 6 describe a ten-fold comparison
with other lightweight networks, reporting RegNetY-400MF at mean 89.87 and SD 1.28.
The method describes taking best validation accuracies across folds. Preserve that
description rather than treating it as our untouched final test evaluation.
Table 7 reports 0.40 GFLOPs, 16.8 MB and 4,344,144 parameters.

**Critical reading:** The authors defer actual edge-device inference timing to
future work. A compact model and GFLOP count do not establish measured latency
on our laptop or a phone. The inspected methods do not establish that their
duplicate handling matches our audit. We cannot directly compare our cleaned
hold-out score to their published cross-validation summary.

**Use in our project:** Add this paper to the literature survey and treat it as a
stronger reference for compact models. If time permits, discuss adding RegNetY as
an additional experimental baseline. That is a proposed scope extension, not a
replacement already made to our approved plan.

**Read:** Sections 2.1–2.2, 3.1–3.7, Tables 5–7, Discussion and Conclusions.

## P2 — Direct precedent for weighted training

**Jlassi, A., Elaoud, A., Ghazouani, H., & Barhoumi, W. (2024). Potato Leaf Disease
Classification Using Transfer Learning and Reweighting-Based Training with
Imbalanced Data.** *SN Computer Science, 5*, 987.
[Publisher / DOI](https://doi.org/10.1007/s42979-024-03334-x).

The publisher abstract describes MobileNetV2 fine-tuning, explainability,
augmentation and reweighting on PlantVillage, with 98.61% accuracy. This directly
motivates our class-weighting ablation; reweighting itself is not our invention.
**Access limit:** only the abstract was verified. Obtain institutional full text
before describing its exact weighting formula, class selection or split procedure.
Its score is not a seven-category Mendeley benchmark.

## P3 — Same dataset, more complex architecture

**Mondal, A., Chatterjee, A., & Avazov, N. (2026). A hybrid CNN-transformer model
with adaptive activation function for potato leaf disease classification.**
*Scientific Reports*. Published **6 January 2026**.
[Publisher / DOI](https://www.nature.com/articles/s41598-025-34406-4).
DOI: **10.1038/s41598-025-34406-4**. The DOI's “025” does not make it a 2025 publication.

**Plain explanation:** Combine a CNN's local image features with a Transformer's
ability to relate distant image regions, and adjust the activation function during
learning.

**Method:** PLDNet uses an ImageNet-pretrained DenseNet169 backbone, a Transformer
component and an adaptive AFpM activation. The authors train separately on
PlantVillage and Mendeley, using 80:20 stratified train/test splits. Their
statistical analysis reports five independent random-split experiments. The
experimental-setup section also uses validation-based callbacks; the independent
validation allocation should be checked before replication.

**Results:** The statistical section reports average accuracies of 99.54% on
PlantVillage and 87.50% on Mendeley, with stated SDs of 0.18% and 0.47%. The model
description gives approximately 32 million trainable parameters.

**Limitations:** The authors explicitly state that they did not train on one
dataset and test on the other. Strong scores on separately trained datasets do
not establish cross-domain transfer. They also discuss computational overhead and
difficult field conditions. Their code-availability statement says code will be
made public on acceptance; this review did not verify a working public repository.

**Use in our project:** Supports testing a simpler model's performance/cost
trade-off and clearly stating generalisation limits. It does not require us to
implement a Transformer or claim that field robustness has already been solved.

**Read:** Dataset description, Data pre-processing, Proposed model, Experimental
setup, Statistical analysis, Limitations and future work.

## P4 — Training strategies and ensemble learning

**Jain, A., et al. (2026). Optimized CNN-based ensemble deep learning approach
for potato leaf disease detection with data augmentation.** *Scientific Reports*.
Published **18 May 2026**.
[Publisher / DOI](https://www.nature.com/articles/s41598-026-50480-8).
DOI: **10.1038/s41598-026-50480-8**.

**Plain explanation:** Train CNNs using different update rules and combine their
decisions, while addressing an underrepresented category using augmentation.

**Method and data:** Three PlantVillage potato classes: early blight, late blight
and healthy. Source counts are 1,000, 1,000 and 152. The ensemble combines CNNs
trained with Adam, SGD, RMSprop and Adamax. The paper describes an 80:20 split
with a validation allocation taken from training. EDL10 reports 97% accuracy.

**Critical reading:** The described pipeline places augmentation/balancing before
dataset splitting in parts of the text. Before replication, verify whether images
derived from one original remain together. This is an evaluation concern, not
proof that leakage occurred. Keep our rule: split originals/groups first and
augment training images only. Its three-category score is not directly comparable
with our seven-category task, and ensemble deployment costs must include all members.

**Use in our project:** Motivates controlled optimizer or augmentation comparisons
if needed later. Do not immediately add a four-model ensemble to our core scope.

**Read:** Data preparation and augmentation, Proposed model, Results on imbalanced
and balanced data, Conclusion; check the origin/split independence carefully.

## P5 — A non-deep-learning imbalance comparison

**Talab, H. K., Mohammadzamani, D., & Parashkoohi, M. G. (2024). Investigating the
accuracy of classification in unbalanced data in order to diagnose two common
potato leaf diseases (early blight and late blight) using image processing and
machine learning.** *Discover Applied Sciences*. Published **22 May 2024**.
[Publisher / DOI](https://link.springer.com/article/10.1007/s42452-024-05959-2).
DOI: **10.1007/s42452-024-05959-2**.

**Plain explanation:** Extract colour, texture and shape information, then study
whether balancing the examples helps a Random Forest classify leaves.

**Data and method:** A separately collected 1,900-image dataset with 550 healthy,
350 early-blight and 1,000 late-blight images; the paper specifies 80% training and
20% testing. It compares the original distribution with over- and undersampling,
using image features, Random Forest and Relief feature selection.

**Results:** Reported accuracies are 87.89%, 91.09% and 96.17% for original,
oversampled and undersampled scenarios, respectively.

**Critical reading:** These findings concern a different collection and three
classes. Check whether resampling changes test composition before attributing all
of an accuracy difference to better learning. Undersampling can discard valuable
examples, so do not remove our majority-category images simply to copy this result.

**Use in our project:** Supports an explicit imbalance experiment and per-class
metrics rather than reliance on overall accuracy alone. It is supporting evidence,
not a reason to replace the custom CNN with Random Forest.

**Read:** Sections 2.1–2.4, 2.6–2.9 and 3.1–3.2.

## P6 — Read the dataset's original article

**Shabrina, N. H., et al. A novel dataset of potato leaf disease in uncontrolled
environment.** *Data in Brief, 52*, 109955.
[Elsevier article](https://www.sciencedirect.com/science/article/pii/S2352340923009861)
and [open full text in PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10733095/).
DOI: **10.1016/j.dib.2023.109955**.

**Date qualification:** The journal issue is **February 2024**, but the online
publication date is **12 December 2023**. Use this essential provenance reference;
if the guide applies 2024–2026 strictly by first-online date, do not count it as a
qualifying recent base paper without confirmation. The Mendeley dataset itself
was released in November 2023, which is separate again.

The paper explains the field collection, labels and dataset construction and
benchmarks several pretrained CNNs. EfficientNetV2B3 reports 73.63% without
augmentation; Table 7 reports 72.35% with augmentation. Its split is 90:10 before
taking a validation subset from training, so its protocol differs from ours.
This is a useful reminder that augmentation is not guaranteed to improve results.
Read collection details, Tables 6–8 and the limitations before discussing why
field images can be harder than PlantVillage.

## How this changes our project justification

### Claims to avoid

- “Existing models only achieve 62.6% on potato.” More relevant published models
  report stronger scores on this collection.
- “Weighted loss is our new algorithm.” P2 and earlier literature already study
  reweighting.
- “A 99% PlantVillage model should get 99% on our data.” Categories, image
  conditions and evaluation protocols differ.
- “Our audit proves the papers' results are wrong.” We have not reproduced their
  experiments; our 49 duplicate exclusions and 12 conflicting-label quarantines
  do not establish the cause of any published score.

### Defensible scope

Keep the seven-category task, audited dataset, InceptionV3 + SVM baseline and
custom-CNN/class-weighting experiments already planned. Present the work as a
**reproducible experimental comparison**, with recorded duplicate handling,
minority-category metrics and measured inference cost. A novel accuracy record
is not promised. Discuss adding RegNetY-400MF as a stronger comparator before
expanding implementation scope.

Suggested mentor explanation:

> We reviewed hybrid feature-classifier methods, compact transfer-learning models,
> imbalance handling and CNN–Transformer methods. Some studies report higher
> performance on the same dataset than our original reference. Our project will
> compare selected methods on a common audited split, test class weighting and
> report prediction quality alongside computational cost. We will disclose that
> our cleaned dataset and protocol differ from published experiments.

## Team reading assignments

| Member | Lead reading | What to bring to the discussion |
|---|---|---|
| Piyush Kumar | P6 dataset article + existing P0 | Dataset provenance, original benchmark, how our audit changes image selection |
| NIKETH P | P1 RegNetY + P3 PLDNet | Architecture sketch, transfer learning vs scratch training, model-cost comparison |
| OMKAR KUMAR | P2 reweighting + P5 resampling | Imbalance strategies, per-class metrics, weighting vs sampling distinctions |
| OMKRRISH | P4 ensemble + P1 deployment discussion | Pipeline diagram, augmentation/split order, complete-system latency requirements |

These are proposed assignments following the existing roles. Everyone should
read P0 and P1. For each assigned paper, record the problem, dataset/classes,
method, split, metrics, reported results, limitations and one usable lesson.

## Verification and search scope

Exa screening covered **31 returned search results across three themes**: the same
seven-category collection, imbalance/field conditions, and lightweight CNNs.
Titles and duplicate results were consolidated; conference papers, unsupported
publisher choices and unrelated articles were not retained as base references.
Publisher/full-text material was read for P1 and P3–P6. P2 is explicitly
abstract-only. Search-result dates were corrected using the article text.

This is a targeted project literature review, not a systematic review or a claim
to have read every cited article in full. No guarantee of experimental validity
or state-of-the-art coverage is implied. The linked articles are the sources;
the limitations and project recommendations above are our critical assessment.
