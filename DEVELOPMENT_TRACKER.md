# Potato Leaf Lab — Development Tracker

Last updated: 23 November 2026 (Asia/Calcutta)

## Project scope

Develop a seven-category potato leaf image classifier. Compare five experiments using shared partitions: InceptionV3 features + SVM, fine-tuned RegNetY-400MF, fine-tuned MobileNetV2, a custom CNN with standard loss, and the same custom CNN with class-weighted loss. Demonstrate the selected model through a Streamlit application.

## Current position

**Dataset preparation and the InceptionV3 + SVM baseline are complete. Custom CNN development is next.** The saved baseline achieved **68.87% validation accuracy** and **63.36% validation macro-F1** on 453 images. The 450-image test partition remains reserved. Formal guide acceptance remains unconfirmed.

| Phase | Status | Completion milestone |
|---|---|---|
| 1. Research and scope | Materials prepared; guide acceptance pending confirmation | Scope, literature survey and design accepted |
| 2. Dataset preparation | Complete, with documented audit limitations | Audited dataset and fixed partitions ready |
| 3. Environment and baseline | Complete | Saved InceptionV3 + SVM with validation results and verified CPU/GPU prediction |
| 4. CNN and transfer-model development | Model definitions added; training pending | RegNetY-400MF, MobileNetV2 and both custom CNN experiments completed |
| 5. Evaluation and model selection | Pending | Selected model evaluated on held-out test data |
| 6. Application integration | Pending; design drafted | Upload-to-prediction workflow works in Streamlit |
| 7. Final testing and submission | Pending | Tested demo, final report and presentation ready |

## 1. Research and scope

Completed work

- [x] Chosen seven-category potato leaf classification as the project scope.
- [x] Selected the primary paper, *Advancing plant leaf disease detection integrating machine learning and deep learning*, Scientific Reports, 2025, DOI 10.1038/s41598-024-72197-2.
- [x] Prepared objectives, research questions and the comparison plan.
- [x] Prepared supporting literature material covering 15 assigned papers.
- [x] Prepared the Review 1 presentation using the college template, with six grouped literature slides.
- [x] Drafted architecture, use case, flow, sequence and class diagrams in the chat.
- [x] Created the project repository and data-preparation code.

Pending work

- [ ] Record guide acceptance of the title, scope, domain and SDG mapping.
- [ ] Fill in guide details and section in the presentation.
- [ ] Finalize the design diagrams as review/report artifacts.

## 2. Dataset preparation

Completed work, recorded in the Phase II report prepared on 2 November 2026

- [x] Downloaded all 3,076 source images from Mendeley Data version 1, DOI 10.17632/ptz377bwb8.1.
- [x] Verified source-file hashes against publisher metadata.
- [x] Checked that all source images decode successfully.
- [x] Excluded 49 redundant same-category copies.
- [x] Quarantined 12 files from six identical-image groups with conflicting labels.
- [x] Retained 3,015 images in a versioned manifest.
- [x] Grouped detected related-image candidates before partitioning.
- [x] Created fixed training, validation and test partitions using seed 42.
- [x] Recorded successful retained-image hash checks and partition-overlap validation.
- [x] Recorded seven passing data-preparation tests.
- [x] Documented dataset attribution, exclusions and reproduction commands.

| Partition | Images |
|---|---:|
| Training | 2,112 |
| Validation | 453 |
| Test | 450 |
| Total retained | 3,015 |

The seven categories are bacteria, fungi, healthy, nematode, pest, phytophthora and virus. The audit does not establish independence of every leaf, plant or capture session. Guide/team review of grouping and quarantine decisions remains open. Any deliberate revision should create a new manifest version and record its reason.

## 3. Environment and baseline implementation

- [x] Set up and record the model-training dependencies.
- [x] Verify PyTorch installation and CUDA execution on the laptop GPU.
- [x] Load images using the existing fixed manifest.
- [x] Implement InceptionV3-specific input preprocessing and frozen feature extraction.
- [x] Fit the SVM using training features and choose its settings using validation data.
- [x] Save the baseline, class-label order and preprocessing configuration.
- [x] Record validation accuracy, macro-F1 and per-category recall.
- [x] Verify saved-bundle image prediction on both CPU and CUDA against the recorded validation prediction.

Milestone achieved on 23 November 2026. PyTorch 2.9.1+cu128 successfully executed on the RTX 3050. The selected RBF SVM uses C=10 and gamma=scale. Validation accuracy is 68.87% and macro-F1 is 63.36%. Eight tests passed. See [Phase III results](PHASE_III_RESULTS.md) and [reproduction commands](BASELINE.md).

## 4. CNN and transfer-model development

| Experiment | Current status |
|---|---|
| InceptionV3 + SVM | Complete; 68.87% validation accuracy and 63.36% macro-F1 |
| RegNetY-400MF | Seven-category definition added; training pending |
| MobileNetV2 | Seven-category definition added; training pending |
| Custom CNN with standard loss | Implementation and training pending |
| Custom CNN with class-weighted loss | Implementation and training pending |

- [x] Add RegNetY-400MF and MobileNetV2 definitions with seven-category output layers.
- [x] Define separate head-training and final-backbone fine-tuning stages, preserving frozen batch-normalization statistics.
- [x] Verify ImageNet-pretrained model loading and seven-category synthetic-input outputs on CUDA; all 10 project tests pass. See [model readiness report](reports/transfer_models.json).
- [ ] Train the RegNetY-400MF head, then fine-tune its final backbone stage using the fixed training/validation partitions.
- [ ] Train the MobileNetV2 head, then fine-tune its final four feature blocks using the same partitions.
- [ ] Record checkpoints, configurations and validation results for both pretrained models.

- [ ] Implement the proposed four-block CNN with 32, 64, 128 and 256 filters.
- [ ] Implement 224 × 224 RGB preprocessing, normalization and training-only augmentation.
- [ ] Train with standard cross-entropy and save the selected validation checkpoint.
- [ ] Train the matched class-weighted variant using weights calculated from training counts.
- [ ] Record configurations, seeds, training curves and validation metrics for both variants.

Milestone: all four neural-network experiments have reproducible training runs and validation results. Select settings using validation macro-F1, alongside accuracy and per-category recall. Keep the test partition reserved for final evaluation.

## 5. Evaluation and model selection

- [ ] Compare all methods on the shared validation partition.
- [ ] Compare macro-F1, accuracy, per-category recall, complete-model storage and batch-1 CPU latency.
- [ ] Choose the deployment model and record the selection rationale.
- [ ] Lock model settings before final test evaluation.
- [ ] Evaluate the selected model on the 450 held-out test images.
- [ ] Produce the confusion matrix, per-category results and error analysis.
- [ ] Document protocol differences when discussing the primary paper's 62.6% potato result.

Milestone: a supported model-selection decision and final test report. Achieving a particular accuracy remains an experimental outcome.

## 6. Application integration

Proposed application structure: browser interface → Python prediction service → saved classifier.

- [ ] Build the Streamlit image-upload interface.
- [ ] Implement image validation and clear invalid-upload feedback.
- [ ] Load the selected model with its matching preprocessing and class labels.
- [ ] Connect upload, preprocessing, inference and category-name display.
- [ ] Verify that application predictions match direct model inference.

Milestone: a user can upload a valid image and receive its predicted category through the complete application.

## 7. Final testing and submission

- [ ] Test supported image uploads, invalid files and repeated predictions.
- [ ] Confirm the project runs from the documented setup on a clean environment.
- [ ] Finalize architecture and UML diagrams to match the implementation.
- [ ] Complete the report with methods, results, limitations and references.
- [ ] Update the presentation with actual experiment results.
- [ ] Prepare demonstration images and rehearse the team presentation.

Milestone: a reproducible application, complete report and presentation ready for the applicable review/submission.

## Evidence and file locations

The original project folder is `C:/Users/Lenovo/.codex/visualizations/2026/10/26/01a1786e-86f2-7302-8d81-eebdce12c1bb/potato-leaf-lab`. Continue all development and tracker updates in this existing checkout. It contains the source code, local raw dataset, audit reports, research documents and latest presentation. GitHub contains the source code, tracker, reports and review documents. Raw dataset files remain local under the existing dataset exclusion rule.

- [Phase I scope document](PHASE_I_RESEARCH_AND_SCOPE.md)
- [Phase II results](PHASE_II_RESULTS.md)
- [Phase III baseline results](PHASE_III_RESULTS.md)
- [Baseline reproduction and image prediction](BASELINE.md)
- [Dataset preparation and reproduction](DATASET.md)
- [Fixed dataset manifest](reports/data-v1/manifest.csv)
- [Review 1 presentation](docs/presentations/Plant_Leaf_Review1_Literature_Updated.pptx)
- [GitHub repository](https://github.com/KrPIYUSH007/potato-leaf-lab)

## Updating this tracker

Update this file after each development task with the date, completed checklist items, supporting files or results, remaining work and the next milestone. Check items only when the work is verified. Record blockers and changed decisions here. Update the local file and sync its changes to the GitHub repository after completed project tasks in this chat or when explicitly requested.

## Progress log

| Date | Verified progress | Next step |
|---|---|---|
| 2 November 2026 | Phase I planning package and Phase II dataset results prepared, as recorded in their reports | Configure model-training environment |
| 5 November 2026 | Revised Review 1 deck produced with 15 supporting papers across six literature slides | Complete design and begin baseline implementation |
| 11 November 2026 | Architecture and UML diagrams drafted in the chat, including a three-layer prediction architecture | Finalize diagrams against implementation |
| 23 November 2026 | Inspected saved preparation reports and created this tracker. Model-training files/results remain pending in the inspected checkout | Verify PyTorch/CUDA and implement InceptionV3 + SVM |
| 23 November 2026 | Synced the tracker and review documents into the original potato-leaf-lab checkout and existing GitHub repository | Begin baseline implementation |
| 23 November 2026 | Completed Phase III in the original checkout. Verified CUDA, trained eight SVM candidates, saved the selected model and validation reports, passed eight tests, and checked saved-model prediction on CPU and CUDA. Accuracy 68.87%, macro-F1 63.36% | Develop and train the custom CNN with standard and class-weighted loss |
| 23 November 2026 | Added RegNetY-400MF and MobileNetV2 to the comparison scope and implemented their seven-category definitions with staged fine-tuning | Train the additional models and implement the custom CNN variants |
