# Phase I — Research foundation and project scope

**Project:** Plant Leaf Disease Classification Using a Custom CNN  
**Focused task:** Seven-category potato leaf image classification  
**Prepared:** 2 November 2026  
**Status:** Research and planning package prepared; team and guide acceptance pending. No models trained or project accuracy measured.

## 1. Project in one paragraph

We will study a published hybrid deep-learning and machine-learning method, implement an InceptionV3 feature extractor with an SVM classifier, and compare it with a custom CNN trained from scratch on the paper's referenced potato dataset. We will test ordinary and class-weighted cross-entropy training to investigate category imbalance. Both approaches will use common data partitions. Evaluation will cover prediction quality, per-category errors, saved-model storage and CPU inference latency. A Streamlit application will demonstrate image upload and prediction using a trained model. Any improvement remains a hypothesis until evaluated.

## 2. College requirements and our proposed decisions

The supplied documents are requirements for the coursework; they are separate from the team's proposed technical choices below.

| Requirement from supplied documents | How this project addresses it |
|---|---|
| The 23 September 2026 notice specifies four students | Four members identified in Section 11. The older syllabus allows three to four. |
| Review recent 2024–2026 journal papers from specified publishers; conference papers are not permitted as the base | Primary reference is a 2025 Scientific Reports journal article from Springer Nature. Supporting journal articles are listed in Section 4. |
| Study methods, algorithms, findings and limitations; identify a feasible gap | Paper summary, method distinction, bounded research question and experiment plan follow. |
| Syllabus requires implementation of algorithms and technologies from the paper | Implement the selected InceptionV3 + SVM approach, in addition to the proposed CNN. |
| Map domain and SDG with a justification, in consultation with the guide | Proposed domain: AI/ML and computer vision. Proposed SDG: Goal 2, target 2.4, through investigation of image-based crop-health monitoring. This is a potential contribution, not a demonstrated agricultural impact. |
| Review 1 is literature survey; subsequent reviews cover design, implementation and demonstration | This package supplies research and scope material. It does not claim implementation progress. |
| Title, domain and SDG require guide consultation | Prepared for discussion; no approval is claimed. |

The notice schedules Review 1 for 5–9 October, Review 2 for 27–31 October, Review 3 for 23–30 November, and signed report submission by 15 December 2026. These are the original dates, not a revised schedule. This package is prepared on 2 November; confirm the applicable review arrangement with the guide. The notice specifies report/PPT similarity and AI-generated-content thresholds below 25%, and the syllabus's exemplary plagiarism band is at most 20%. This document has not undergone institutional checks; students must understand, verify and write their submission according to college policy.

**Our proposals, not college-mandated choices:** potato-only scope; PyTorch; a four-block CNN; class-weighted loss; initial 70/15/15 partitions; Streamlit. These choices can be adjusted through guide consultation.

## 3. Primary paper study

### Identification and provenance

**Sujatha, R., Krishnan, S., Chatterjee, J. M., and Gandomi, A. H.** “Advancing plant leaf disease detection integrating machine learning and deep learning.” *Scientific Reports*, volume 15, article 11552, published 4 April 2025. DOI: **10.1038/s41598-024-72197-2**. [Publisher article](https://www.nature.com/articles/s41598-024-72197-2).

This paper was selected during project discussion with AI assistance and checked against the publisher article. Do not say that this exact title is prescribed in the guidelines: the supplied guidelines state paper-selection criteria rather than naming this paper.

### Problem and method

The study investigates image classification for several crops using pretrained VGG19 and InceptionV3 to extract features, followed by classifiers including SVM, kNN, random forest, decision tree and AdaBoost. Its central pipeline combines a deep feature representation with a separate traditional classifier. For our potato baseline, the relevant path is **image → InceptionV3 features → SVM → predicted category**. The paper evaluates combinations with stratified 10-fold cross-validation.

### Selected result and interpretation

For potato, the authors report best performance with InceptionV3 + SVM: **62.6% accuracy**, 63% precision, 62.6% recall, 62.1% F1, 89% AUC and 54.2% MCC. Preserve the authors' metric labels; do not relabel their F1 as macro-F1 without verifying the averaging convention. These are published results, not our measurements. See the article's Abstract, Results and Potato leaf dataset discussion.

### Limitations and questions

- The authors identify further validation with newly collected field leaves, field studies, and consultation with agricultural scientists and farmers as future work.
- The dataset has variable capture conditions and unequal class counts. Their influence on this particular result must be tested; they do not by themselves prove why accuracy was 62.6%.
- A custom CNN is an alternative worth testing, not an established improvement.
- Before calling our baseline an exact replication, resolve implementation details: input preprocessing, feature-extraction layer, feature scaling, SVM kernel and settings, tuning procedure, and fold assignments. The passages reviewed do not provide a complete executable configuration. Record every implementation choice that cannot be recovered from the paper or supplementary material.
- Class weighting and CNNs are established methods. Our contribution is a controlled implementation and comparison on this task, not a claim that these techniques are new.

## 4. Supporting literature and bounded research gap

The following supporting papers were used for the existing Review 1 deck. Their scores are context; different datasets and protocols prevent direct numerical ranking against our experiment.

| Reference | Relevance to this project | Reading status |
|---|---|---|
| Ugwah, V. N. and Abolghasemi, V. “Optimised MobileNet for very lightweight and accurate plant leaf disease detection.” Scientific Reports 15, 43690 (2025). [DOI](https://doi.org/10.1038/s41598-025-27393-z) | Motivates measuring compactness together with predictive performance. Does not require us to implement MobileNet. | Publisher material verified during presentation preparation; team should read methods. |
| Duhan, S., Gulia, P., Gill, N. S. and Narwal, E. “RTR_Lite_MobileNetV2: A lightweight and efficient model for plant disease detection and classification.” Current Plant Biology 42, 100459 (2025). [DOI](https://doi.org/10.1016/j.cpb.2025.100459) | Supports investigating performance across differing image conditions and datasets. | Publisher material verified during presentation preparation; team should read methods. |
| Arya, A. and Mishra, P. K. “MobileNetV2-Incep-M: a hybrid lightweight model for the classification of rice plant diseases.” Multimedia Tools and Applications 83, 79117–79144 (2024). [DOI](https://doi.org/10.1007/s11042-024-18723-w) | Provides architecture-design context; rice evidence does not establish potato performance. | Abstract-level verification only; obtain full text before making detailed methodological claims. |

**Bounded gap:** For our selected task, we will investigate whether an end-to-end custom CNN offers a useful prediction-quality and computational-cost trade-off relative to an InceptionV3 + SVM baseline, and whether class weighting improves minority-category recognition. This is a project-specific question. A 62.6% result does not establish an unsolved problem across all plant-disease research.

## 5. Problem statement and objectives

**Problem statement:** Develop and evaluate a reproducible seven-category potato leaf classifier on uncontrolled-environment images, comparing a paper-based hybrid baseline with a custom CNN while accounting for category imbalance and computational cost.

**Research question:** On common audited data partitions, how do InceptionV3 + SVM, an ordinary-loss custom CNN, and a class-weighted custom CNN compare in macro-F1, category recall, accuracy, model storage and CPU latency?

**Objectives:**

1. Audit the referenced dataset and save reproducible partitions with leakage checks.
2. Implement the selected InceptionV3 + SVM method and document reproduction differences.
3. Train the proposed custom CNN from scratch using training-only augmentation.
4. Compare ordinary and class-weighted cross-entropy under otherwise matched settings.
5. Evaluate all approaches on shared held-out data and document errors and limitations.
6. Demonstrate a saved trained model through a Streamlit image-upload application.

**Hypotheses to test:** Class weighting may improve macro-F1 or minority recall; the custom CNN may require less storage or lower latency than the complete baseline. Either can be false, and weighted loss can reduce other categories' performance.

## 6. Dataset choice

**Potato Leaf Disease Dataset in Uncontrolled Environment**, Mendeley Data, version 1, published 10 November 2023. DOI: **10.17632/ptz377bwb8.1**. [Dataset page](https://data.mendeley.com/datasets/ptz377bwb8/1). This is reference 26 in the primary article. The repository describes 1,500 × 1,500 JPG images captured using smartphone cameras on potato farms in Central Java, Indonesia. Licence: **CC BY 4.0**; preserve attribution and document modifications.

| Category | Images reported in primary paper |
|---|---:|
| Bacteria | 569 |
| Fungi | 748 |
| Healthy | 201 |
| Nematode | 68 |
| Pest | 611 |
| Phytophthora | 347 |
| Virus | 532 |
| **Total** | **3,076** |

These are broad dataset categories, not seven individually named pathogens. Predict the supplied label rather than claiming identification of a precise causal organism. Counts above are reported values, not a completed local audit. A majority-class-only predictor would score about **24.3%** on the reported full distribution (748/3076); calculate the actual reference on our test distribution later.

Phase II must verify the downloaded version, folders, labels, image integrity, decoded-pixel duplicates and near-duplicates. Inspect related photographs and available plant/session metadata. Keep related images together across partitions where identifiable. If metadata is absent, record that exact and visual duplicate checks cannot guarantee plant-independent evaluation. Do not augment before splitting.

## 7. System scope and architecture

**Core scope:** One uploaded image at a time; seven-category classification; custom CNN; paper-based baseline; class-weighting ablation; measured comparison; local Streamlit demonstration.

**Outside the core scope:** All-crop recognition, precise pathogen confirmation, disease severity estimation, treatment advice, live video, farmer-scale deployment and guaranteed unfamiliar-image rejection. Grad-CAM or additional field testing may be added after the core works, if time and suitable data allow.

```text
Research and dataset audit
          |
Shared train / validation / test manifests
          |
          +-- InceptionV3 features --> scaler --> SVM
          |
          +-- Custom CNN --> ordinary cross-entropy
          |
          +-- Same custom CNN --> class-weighted cross-entropy
          |
Shared evaluation --> reports, confusion matrices, model artifacts
          |
Selected trained CNN --> Streamlit image upload and prediction
```

Proposed CNN starting point: 224 × 224 RGB input; convolution blocks with 32, 64, 128 and 256 filters; batch normalization, ReLU and downsampling; global average pooling; dropout; seven output logits. Train from scratch. This is a proposal rather than a frozen architecture or new algorithm. Follow weight-specific preprocessing for the baseline; InceptionV3 normally uses 299 × 299 input, so preprocessing must be recorded separately for each model.

## 8. Predeclared experiment plan

| ID | Configuration | Purpose |
|---|---|---|
| B0 | Training-majority category predictor | Simple context for interpreting accuracy |
| B1 | Frozen pretrained InceptionV3 features + training-fitted scaler + SVM | Selected paper-based method |
| C1 | Custom CNN + ordinary cross-entropy | Main alternative |
| C2 | Same CNN + class-weighted cross-entropy | Isolate weighting effect |

### Data and selection rules

- Initial experiment: approximately 70% training, 15% validation, 15% testing, stratified where compatible with grouping. Group integrity takes priority over exact percentages. Save the manifest, hashes, seed, counts and exclusions; use the same manifest for all methods.
- Derive scaling, weights and any learned preprocessing from training data only. Augment training images only. Preserve independent deterministic validation/test preprocessing.
- CNN starting settings: Adam, learning rate 0.001, batch size 32 if memory permits, at most 50 epochs, dropout 0.35 and early-stopping patience 8. Select checkpoint by validation macro-F1. These are our initial settings, not the paper's recovered settings.
- Proposed class weights: N/(K × n_c), using training counts only. For the C1/C2 comparison keep architecture, augmentation, initialization seed and other training settings matched.
- Baseline starting search: SVM kernel in {linear, RBF}, C in {0.1, 1, 10}; RBF gamma in {scale, 0.001, 0.01}. Fit candidates using training features and select using validation macro-F1. Save chosen settings and any departure from recovered paper details.
- Initial seed: 42. If compute permits, repeat locked training settings with seeds 7 and 21 on the same fixed split and report all runs, not just the best test score.
- Choose final settings from validation results before test evaluation. Do not use test errors to retune and then report the same set as an untouched final evaluation.

### Metrics and practical measurements

Primary selection metric: **macro-F1**. Report accuracy, macro precision/recall, per-category precision/recall/F1/support, and a confusion matrix with fixed label ordering. Show whether minority-category gains come at other categories' expense. Retain individual predictions so results can be inspected.

For CPU timing, fix hardware, thread count and batch size 1; warm up each model and report median and 95th-percentile latency over repeated measurements. Time the full baseline feature extractor + scaler + SVM, rather than the SVM alone. Distinguish model-only and image-to-result timing. Count all inference artifacts in storage, including InceptionV3 weights; record file format and parameter count where applicable.

### Comparison with the published number

The paper reports stratified 10-fold cross-validation. Our first held-out experiment is an **adapted comparison**, not an exact reproduction. Exceeding 62.6% on one split does not prove improvement over the paper. Compare B1/C1/C2 directly within our protocol. If resources and guide expectations support a closer replication, implement common 10-fold outer splits with training-only inner validation for model selection, and document remaining differences. Do not make this expensive extension a prerequisite for the first working system.

### Definition of a successful project

Success means a working, reproducible system with a fair and interpretable comparison, not a promised accuracy. A claim of higher accuracy requires measured evidence under a common protocol; a claim of a better trade-off must state the actual performance and latency/storage changes. Report negative results honestly. No target such as 90% is guaranteed.

## 9. Tools and resources

| Purpose | Proposed tools |
|---|---|
| Development | Python 3.11/3.12, virtual environment, VS Code, Git/GitHub |
| CNN and pretrained features | PyTorch, Torchvision |
| SVM and metrics | Scikit-learn |
| Image and tabular processing | Pillow, NumPy, Pandas |
| Graphs | Matplotlib, Seaborn |
| Demonstration | Streamlit |
| Training | Local or cloud GPU if available; CPU for smoke tests and inference |

Pin compatible package versions after verifying the environment. Model design does not require a paid AI API. Hardware, GPU availability and storage budget have not yet been verified. Dependency installation and dataset download are Phase II tasks, not completed Phase I work.

## 10. Application requirements

The application must load a saved checkpoint with its class mapping and preprocessing, accept a readable image, display a preview and predicted supported category, and handle invalid uploads. Present any softmax score as a model confidence estimate, not a verified probability of disease. If an uncertainty threshold is used, choose and assess it on validation data. A seven-class model cannot be assumed to reject non-potato or unfamiliar images; communicate supported input and evaluation limits. Actual unfamiliar-image rejection requires additional data and validation.

## 11. Team and proposed responsibilities

The following allocation follows the earlier four-role plan and provided name order; it is a proposed allocation for team confirmation.

| Member | USN | Phase I task | Later technical ownership |
|---|---|---|---|
| Piyush Kumar | 1BF24CS216 | Summarise primary paper, verify dataset reference, consolidate literature notes | Dataset audit, validation and manifests |
| NIKETH P | 1BF24CS199 | Explain feature extraction, SVM and proposed CNN architecture | CNN training and saved checkpoints |
| OMKAR KUMAR | 1BF24CS205 | Check metrics, reported results and experimental limitations | Baseline, evaluation and weighting ablation |
| OMKRRISH | 1BF24CS204 | Document system modules, application flow and repository organisation | Streamlit, integration and end-to-end testing |

All members should read the primary paper, understand both approaches, review another member's work, and help prepare and explain the presentation. No member is limited to documentation alone.

**Guide:** to be supplied. **Designation:** to be supplied. **Section:** to be supplied. Approval is not recorded.

## 12. Phase I completion checklist

### Prepared in this package

- [x] Identified and verified primary journal paper and dataset reference.
- [x] Summarised relevant method, potato result and limitations.
- [x] Consolidated supporting literature with reading-status caveats.
- [x] Defined problem, objectives, bounded scope and hypotheses.
- [x] Specified architecture, comparison methods and experiment rules.
- [x] Listed resources, proposed responsibilities and Phase II handoff.
- [x] Distinguished coursework requirements, published findings and proposed work.

### Pending human decisions or later evidence

- [ ] Each member reads the primary paper and explains their assigned section.
- [ ] Team confirms responsibilities and initial experiment choices.
- [ ] Guide discusses/approves title, scope, domain, SDG and baseline adaptation.
- [ ] Record guide details, section, meeting date and any requested changes.
- [ ] Resolve baseline reproduction details from available paper materials; log unresolved choices.
- [ ] Obtain full text of abstract-only supporting reference if retained for detailed analysis.

Phase I preparation is complete as a discussion package. Formal academic approval and demonstrated team understanding cannot be marked complete by this document.

## 13. Mentor discussion script

**What are you building?** A seven-category potato leaf classifier with a custom CNN, compared with the paper's InceptionV3 + SVM method.

**What is your modification?** We replace the separate pretrained-feature/SVM pipeline with an end-to-end CNN trained from scratch and test a class-weighting ablation. The resulting improvement is a hypothesis, not an existing result.

**Why this dataset?** It is referenced by the primary paper, publicly available with attribution, and contains variable field-image conditions. Its imbalance makes per-category evaluation meaningful.

**Why did the paper get 62.6%?** That is its reported result. We have not isolated its causes. Data imbalance, visual overlap and capture variation are factors to examine rather than proven explanations.

**Will you beat it?** We cannot promise that. We will compare both methods under common conditions and report the actual results. A different split cannot establish superiority over their published cross-validation result.

**Is the work completely new?** It is a paper-based implementation with an experimental extension. CNNs, SVMs and weighted loss already exist.

**Does Streamlit improve the model?** No. It is the demonstration interface; predictive improvements must come from measured model or training changes.

**How does this satisfy the guidelines?** We study a recent qualifying journal article, implement a selected algorithm from it, define an engineering question and evaluate a proposed extension. Final acceptability is for the guide to determine.

## 14. Phase II handoff

Next technical milestone: verify the Python environment, download the cited dataset version, inspect its actual structure, create an audit report and frozen shared manifests, and run a small training/inference smoke test. Do not train on fabricated placeholder images and report the resulting accuracy as project performance. The current repository is planning material, not a trained project.

## Sources

1. Primary research article: [Sujatha et al., Scientific Reports (2025)](https://www.nature.com/articles/s41598-024-72197-2). Consult Abstract, Material and methods/Dataset, System components, Results/Stratified 10-Fold cross validation, potato results, Conclusion and Data availability.
2. Dataset: [Mendeley Data version 1](https://data.mendeley.com/datasets/ptz377bwb8/1), DOI 10.17632/ptz377bwb8.1.
3. Supporting journal references are linked in Section 4; see the stated reading-status limitations.
4. Supplied coursework notice: MINI Project Guidelines 2026-5th semester.pdf, dated 23 September 2026, page 1.
5. Supplied syllabus: mini project syllabus.pdf, pages 1–4 (printed pages 141–144).
6. Proposed SDG context: [UN Goal 2](https://sdgs.un.org/goals/goal2), especially target 2.4. Mapping requires guide discussion.
