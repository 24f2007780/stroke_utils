# Stroke Multimodal Tutorial Series

Back to [Top](../README.md) | [Documentation](../docs/README.md) | [Data](../data/README.md)

This directory contains a **comprehensive, 9-part tutorial and analysis series** guiding researchers step-by-step through multi-modal human stroke analysis (case **HB02CV**): from raw NIfTI MRI coordinate geometry and diffusion physics to digital pathology cellular quantification, peri-infarct gradient modeling, and virtual IHC prediction.

---

## Tutorial Series Overview

The tutorial series is structured as a continuous scientific narrative:

```text
01. MRI QC & Ischemic Core Segmentation
         ↓
02. Longitudinal MRI & Atlas Localization
         ↓
03. 3D MRI → Histology Block Mapping
         ↓
04. Histology ROI Preprocessing & Digital Pathology Annotations
         ↓
05. End-to-End Multimodal Integration & Utility Gaps
         ↓
06. Pathology Boundary Distance Profiler
         ↓
07. IHC Stain Quantification vs Annotations
         ↓
08. Peri-Infarct Protein Gradients
         ↓
09. Virtual IHC from Routine Stains (Nissl & H&E)
```

Every code cell from the original foundational notebooks is preserved verbatim, paired with clear, tutorial-oriented documentation following `/ref_func` style.

---

## Tutorial Curriculum

| Part | Notebook | Core Concept | Key Functions Tested | Key Outputs / Artifacts |
| :---: | :--- | :--- | :--- | :--- |
| **01** | [`01_mri_qc_and_lesion_segmentation.ipynb`](01_mri_qc_and_lesion_segmentation.ipynb) | Single-modality MRI QC (geometry, intensity traps, Otsu brain masking) and dual-modality diffusion physics ($DWI \ge 150, ADC \le 750$), distinguishing cytotoxic core from T2 shine-through. | `load_mri`, `inspect_mri`, `intensity_qc`, `brain_mask`, `qc_summary`, `check_spatial_alignment`, `load_diffusion_pair`, `plot_joint_diffusion`, `restricted_diffusion_lesion`, `show_diffusion_pair`, `paired_threshold_sensitivity` | Complete geometry table, multiplanar orthogonal slice views, 3-panel intensity profiles, whole-brain baseline correlations, 2D joint density plots, multiplanar red/gold contours, and 2D threshold sensitivity grids. |
| **02** | [`02_longitudinal_mri_and_atlas_localization.ipynb`](02_longitudinal_mri_and_atlas_localization.ipynb) | Longitudinal stroke comparison across timepoints (16 dpm acute vs 11 dpm subacute) and localization of lesion burden into MNI 125 structures and JHU 32 arterial territories. | `load_mri`, `brain_mask`, `lesion_mask`, `compare_mri_volumes`, `load_manifest_volume`, `regional_lesion_burden` | Acute-to-subacute volume change ($-13.26$ mL single-threshold, $-49.6\%$ core), Dice overlap, multiplanar visual comparisons, top MNI structures (Putamen $49.1\%$), and top arterial territories (Left MCA occipital/parietal). |
| **03** | [`03_mri_to_histology_mapping.ipynb`](03_mri_to_histology_mapping.ipynb) | Microtome cutting plane geometry (`manifest["blocks"][...]["krow"] = [a, b, c, d]`), projection of 3D lesion voxels onto physical specimen blocks, and extraction of oblique MRI slices matching histology planes. | `map_voxel_to_histology_section`, `extract_block_plane_from_mri` | Block ranking table (Block 313: $46.6\%$ of lesion core, sections ~24–1842), millimeter cutting spans, and resliced 2D oblique MRI appearance matching histology microtome angles. |
| **04** | [`04_histology_annotations_and_pathology.ipynb`](04_histology_annotations_and_pathology.ipynb) | Inspection and preprocessing of 3D histology ROI stacks (cropping, normalization, PyTorch tensors), OpenAtlas GeoJSON annotation querying (CD68, GFAP, FIB, CD34, HIF1a), coordinate transforms, visual overlays, morphological analysis, and tissue validation. | `load_roi`, `preview_sections`, `crop_roi`, `normalize_roi`, `to_torch`, `find_section_anomalies`, `load_stroke_annotations`, `to_roi_coordinates`, `create_annotation_overlay`, `compute_annotation_properties`, `filter_annotations`, `validate_roi_annotations`, `compute_annotation_overlap`, `create_stack_annotation_masks` | Real 117-section Level 4 ROI stack, multi-section previews, ML-ready PyTorch tensors, GeoJSON polygon overlays on stained tissue, per-protein morphological tables, and 3D volumetric annotation masks. |
| **05** | [`05_multimodal_pathology_integration.ipynb`](05_multimodal_pathology_integration.ipynb) | Rapid end-to-end multi-modal validation and single-call pipeline orchestration (`summarize_mri_pathology`) bridging MRI lesion segmentation, specimen block ranking, and digital pathology density profiling; library utility gaps roadmap. | `load_diffusion_pair`, `restricted_diffusion_lesion`, `summarize_lesion_regions`, `compare_lesion_across_mri`, `find_lesion_histology_sections`, `summarize_section_annotations`, `compare_lesion_annotation_density`, `summarize_mri_pathology` | Unified multi-modal bundle summary; cross-scale translation table connecting diffusion restriction to cellular pathology; prioritized library extension roadmap. |
| **06** | [`06_pathology_boundary_distance_profiler.ipynb`](06_pathology_boundary_distance_profiler.ipynb) | Automated marker-vs-lesion boundary gradient quantification across physical microtome blocks, tissue-normalized density profiling, and Detection Studio OpenLayers layer export. | `find_available_annotations`, `fetch_roi_annotations`, `measure_marker_burden`, `compute_boundary_distance_profile` | Empirical availability matrix (1,630 checks), distance gradient profiles (113 rows), compartment summary, and viewer-compatible GeoJSON spatial layer. |
| **07** | [`07_ihc_stain_quantification_vs_annotations.ipynb`](07_ihc_stain_quantification_vs_annotations.ipynb) | Whole-slide immunohistochemistry DAB stain quantification versus expert pathology annotation grades; evaluates whether optical density of DAB alone is sufficient to reproduce pathologist annotations or if cellular morphology and tissue context dominate. | `load_slide_catalog`, `find_slides`, `fetch_region`, `tissue_mask`, `tissue_bbox`, `stain_separation`, `tile_features`, `tile_labels`, `load_annotations`, `annotation_classes`, `rasterize_annotations`, `draw_annotation_outlines`, `scale_bar` | Whole-slide annotation grade distributions, DAB optical density histograms across severity grades, ROC curves and AUC metrics (GFAP, CD68, HIF1a, etc.), spatial agreement/disagreement maps (stain-only vs annotation-only), and multi-protein benchmark table. |
| **08** | [`08_peri_infarct_protein_gradients.ipynb`](08_peri_infarct_protein_gradients.ipynb) | Spatial organization of post-stroke cellular response along distance bands relative to the H&E infarct boundary; pairs registered IHC marker sections (GFAP, CD68, CD34, Fibrinogen, HIF1a, etc.) with neighboring H&E infarct outlines within matched sets to quantify peri-infarct concentration profiles. | `load_slide_catalog`, `slide_set`, `fetch_region`, `tissue_bbox`, `tissue_mask`, `stain_separation`, `load_annotations`, `rasterize_annotations`, `draw_annotation_outlines`, `scale_bar` | Signed distance transform from H&E infarct border (core < 0, peri-infarct border ~0, periphery > 0), distance-resolved coverage profiles in 0.5 mm radial bands, mean DAB optical density gradients, stratified bootstrap uncertainty intervals, and multi-marker spatial peak alignment ranking. |
| **09** | [`09_virtual_ihc_from_nissl_and_he.ipynb`](09_virtual_ihc_from_nissl_and_he.ipynb) | Machine learning predicting IHC protein annotations from cheap routine histology stains (Nissl and H&E) using foundation model representations (e.g. Prov-GigaPath / pathology ViT) and linear probing under leave-one-block-out cross-validation. | `load_slide_catalog`, `slide_set`, `fetch_region`, `tissue_bbox`, `tissue_mask`, `load_annotations`, `rasterize_annotations`, `tile_reduce`, `fetch_thumbnail`, `draw_annotation_outlines`, `scale_bar` | Multi-stain matched tile dataset (IHC label, Nissl, H&E), colour feature baselines vs foundation model embeddings, leave-one-block-out ROC curves and AUC benchmarks across 7 proteins, full-field virtual IHC prediction heatmaps on held-out blocks, and high-confidence / failure mode gallery. |

---

## Production & Cell-Quality Review Audit

The complete cell audit artifact is published in [`../metadata/notebook_cell_review.csv`](../metadata/notebook_cell_review.csv):

- **Total Cells Reviewed:** 140 cells across `MRI_operations.ipynb` (46), `preprocessing_operations.ipynb` (26), `stroke_protein_annotations.ipynb` (52), and `integrated_eda.ipynb` (16).
- **Audit Scores:** Cell-by-cell scores for importance (1–5), redundancy (0–5), and keep/refactor/remove classification with technical rationale.

---

## Execution Guidelines

1. **Working Directory & Python Path**:
   Each tutorial resolves paths relative to the repository:
   ```bash
   cd stroke_notebooks/notebooks
   jupyter notebook
   ```

2. **Data Prerequisites**:
   The tutorial series utilizes real datasets located in [`../data/`](../data/README.md):
   - `DWI-11.nii` (Clinical acquisition)
   - `iv11_DWI.nii.gz` & `iv11_ADC.nii.gz` (Subacute diffusion pair)
   - `iv11_dwi_hyper.nii.gz` (Ground-truth segmentation)
   - `iv16_DWI.nii.gz` (Acute diffusion volume)
   - `16dpm_DWI_regto_exvivo_cropped.nii.gz` & `16dpm_ADC_regto_exvivo_cropped.nii.gz` (Registered ex-vivo pair)
   - `manifest.json`, `lab_mni.bin`, `lab_art.bin` (Reference grid, cutting vectors, atlas labels)
   - `ROI_587_SE1019_X12000_Y12000_Z117_768x768_L4_axial_IHCS.npy` (Real 117-section histology ROI stack, 197 MB)

3. **Methodological Note on MRI-to-Histology Registration**:
   The cutting vector `krow = [a, b, c, d]` in `manifest.json` mathematically resolves the **out-of-plane physical slice coordinate ($Z$ / cutting axis)**. It does **not** solve 2D in-plane pixel registration between histology sections and MRI planes. Tutorials 03 and 05 explicitly document this methodological boundary.

4. **Whole-Slide Streaming & Public C-Stroke Access (Notebooks 07–09)**:
   Notebooks 07–09 stream whole-slide image pyramids, matched multi-stain slide sets, and GeoJSON annotations on-demand from the public C-Stroke platform (`https://c-stroke.humanbrain.in`) via [`slide_utils_v1.py`](../slide_utils_v1.py). Downloaded tiles and metadata are automatically cached under `~/.cache/cstroke/` (configurable via `CSTROKE_CACHE_DIR`).
   - **Dependencies**: In addition to core dependencies, Notebooks 07–09 use `scikit-learn` (ROC/AUC analysis, LogisticRegressionCV) and Notebook 09 optionally uses `torch` and HuggingFace `transformers` (pathology foundation model feature extraction). If `transformers` or GPU acceleration is unavailable, baseline colour features remain fully operational.
