# Stroke Multimodal Analysis & Tutorial Notebooks

Interactive analysis, neuroimaging pipelines, and multimodal data integration for post-mortem human stroke case **HB02CV**.

Online interactive viewer: [https://c-stroke.humanbrain.in/](https://c-stroke.humanbrain.in/)

---

## Quick Links

- Please see [Getting Started] for setup and installation instructions
- Visit [Docs] for architectural documentation and coordinate system references
- Explore the interactive Jupyter [notebooks] demonstrating end-to-end multimodal workflows
- Review the [data] directory for imaging volumes and dataset manifests

[Getting Started]: ./docs/Getting_started.md
[Docs]: ./docs/README.md
[notebooks]: ./notebooks/README.md
[data]: ./data/README.md

---

## Overview & Abstract

This repository provides an integrated analysis toolkit and interactive Jupyter tutorials bridging multi-sequence MRI, ex-vivo reference grids, histology block and section planes, and digital pathology annotations for post-mortem human stroke pathology (case **HB02CV**).

The dataset features:
- **In-Vivo High-Resolution MRI**: DWI and ADC acquisitions across two longitudinal pre-mortem timepoints: **16 dpm** (early post-stroke) and **11 dpm** (evolution).
- **Ex-Vivo Reference Space**: A high-resolution cropped reference volume (211 × 377 × 238 at 0.488 × 0.488 × 0.70 mm).
- **Histology Block & Section Alignment**: Geometric mapping between 3D MRI voxels and physical oblique histology blocks (218, 217, 313, 315, 314) and 20 µm microtome sections across five stain series (HEOS, NISL, MYEL, IHCS, IHC3).
- **Digital Pathology Annotations**: Vectorized multi-protein immuno-histochemical annotations (CD68, GFAP, Fib, CD34, HIF1a) registered via OpenAtlas GeoJSON APIs.
- **Pre-Aligned Anatomical Atlases**: Voxel-aligned 125-region MNI structural labels and 32-region JHU arterial territory masks.

![Production Comparison](./assets/prod.png)

---

## Repository Structure

The repository structure follows the [DHARANI Data Tutorial](https://github.com/SGBC-IITM/DHARANI_data_tutorial/) organization:

```text
stroke_notebooks/
├── README.md                 # Primary project overview and quick links
├── requirements.txt          # Python environment dependencies
├── .gitignore                # Excludes large binaries (*.nii, *.bin), JS, and archives
├── roi_utils.py              # Histology ROI array helper module
├── annotations_utils.py      # OpenAtlas digital pathology annotation module
├── mri_utils.py              # Stroke MRI NIfTI, atlas, and diffusion lesion helper module
├── integrated_utils.py       # Multi-modal bridging and cross-scale integration pipelines
├── pathology_profiler.py     # Pathology-vs-lesion boundary distance and gradient analysis engine
├── slide_utils_v1.py         # Functional WSI, tiling, DAB deconvolution, and public C-Stroke API
├── notebooks/                # Progressive tutorial and analysis notebooks (*.ipynb)
│   ├── README.md
│   ├── 01_mri_qc_and_lesion_segmentation.ipynb
│   ├── 02_longitudinal_mri_and_atlas_localization.ipynb
│   ├── 03_mri_to_histology_mapping.ipynb
│   ├── 04_histology_annotations_and_pathology.ipynb
│   ├── 05_multimodal_pathology_integration.ipynb
│   ├── 06_pathology_boundary_distance_profiler.ipynb
│   ├── 07_ihc_stain_quantification_vs_annotations.ipynb
│   ├── 08_peri_infarct_protein_gradients.ipynb
│   └── 09_virtual_ihc_from_nissl_and_he.ipynb
├── metadata/                 # Quantitative summaries, availability matrices, and audit CSVs
│   ├── annotation_availability_matrix.csv
│   ├── annotation_availability_summary.csv
│   ├── hb02cv_compartment_summary.csv
│   ├── hb02cv_distance_profiles.csv
│   └── notebook_cell_review.csv
├── docs/                     # Architectural documentation and coordinate specs
│   ├── README.md
│   ├── Getting_started.md
│   ├── c-stoke.context.md
│   └── MRIdraft1.md
├── data/                     # Datasets and metadata specifications
│   ├── README.md
│   ├── manifest.json         # Primary reference grid and block affine geometry
│   ├── sets_data.json        # Section and slide set correspondence
│   └── *.nii / *.bin         # Medical imaging data (excluded from Git)
├── assets/                   # Figures, screenshots, and visual assets
│   ├── README.md
│   ├── prod.png
│   └── notebook.png
└── viewer/                   # Standalone and web reference viewer implementations
    ├── README.md
    └── web_reference/        # Web viewer coordinate transform scripts (*.js)
```

---

## Core Helper Modules

The repository provides modular, standalone Python libraries located at the root:

| Module | Scope & Functions |
| :--- | :--- |
| [`roi_utils.py`](roi_utils.py) | **Histology ROI Extraction & Preprocessing**: Loading NumPy stacks from Google Drive / disk (`load_roi`, `load_and_preview_roi`), section previews (`preview_sections`), cropping, grayscale conversion, intensity normalization (`normalize_roi`), and PyTorch tensor conversion (`to_torch`). |
| [`annotations_utils.py`](annotations_utils.py) | **Digital Pathology Annotations**: Querying OpenAtlas GeoJSON vector annotations (`fetch_roi_annotations`), coordinate transformations into ROI pixel space (`to_roi_coordinates`), class/protein filtering (`filter_annotations`), and real-tissue validation against background glass (`validate_roi_annotations`). |
| [`mri_utils.py`](mri_utils.py) | **Stroke Neuroimaging & Multimodal Bridges**: Loading NIfTI volumes with slope/intercept applied (`load_mri`), geometry inspection (`inspect_mri`), lesion segmentation & sensitivity (`lesion_mask`, `restricted_diffusion_lesion`), paired diffusion analysis (`load_diffusion_pair`, `paired_diffusion_stats`), atlas lesion burden (`regional_lesion_burden`), and MRI-to-histology slice extraction (`extract_block_plane_from_mri`). |
| [`integrated_utils.py`](integrated_utils.py) | **Cross-Scale Integration Pipelines**: Anatomical & arterial executive summaries (`summarize_lesion_regions`), longitudinal volume & overlap comparison (`compare_lesion_across_mri`), specimen block projection (`find_lesion_histology_sections`), annotation availability (`summarize_section_annotations`), pathology density quantification (`compare_lesion_annotation_density`), and single-call pipeline orchestration (`summarize_mri_pathology`). |
| [`pathology_profiler.py`](pathology_profiler.py) | **Pathology Boundary Distance Profiler Engine**: Analytical engine for marker-vs-lesion boundary gradient quantification, tissue-normalized density profiling, stratified bootstrap uncertainty, and Detection Studio OpenLayers layer export. |
| [`slide_utils_v1.py`](slide_utils_v1.py) | **Whole-Slide Histology & Public C-Stroke API (v1)**: Streaming IIPImage pyramids (`fetch_region`, `fetch_thumbnail`), tile feature extraction (`tile_features`), stain separation via colour deconvolution (`stain_separation`), annotation rasterization (`rasterize_annotations`, `tile_labels`), slide catalog traversal (`load_slide_catalog`, `find_slides`, `slide_set`), and publication plotting helpers (`draw_annotation_outlines`, `scale_bar`, `set_plot_style`). |
| [`mcp_client.py`](mcp_client.py) | **Model Context Protocol (MCP) Client**: Lightweight interactive MCP client for Google Colab/Jupyter. Connects to local/remote MCP servers, with natural-language requests (`ask_mcp`), tool execution, resource reading, prompt templates, structured schema validation, and inline image rendering. |

---

## Getting Started

```bash
# Clone and install dependencies
git clone https://github.com/yashviup/stroke_utils
cd notebooks
pip install -r requirements.txt

# Launch tutorials
jupyter notebook notebooks/
```

See [docs/Getting_started.md](docs/Getting_started.md) for complete code recipes and tutorial walkthroughs.
