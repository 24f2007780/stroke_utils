# Stroke Multimodal Tutorial Series

Back to [Top](../README.md) | [Documentation](../docs/README.md) | [Data](../data/README.md)

This directory contains a **comprehensive, 5-part tutorial series** guiding researchers step-by-step through multi-modal human stroke analysis (case **HB02CV**): from raw NIfTI MRI coordinate geometry and diffusion physics to digital pathology cellular quantification and unified pipeline orchestration.

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
