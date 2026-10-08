# Tutorial & Analysis Notebooks

Back to [Top](../README.md) | [Documentation](../docs/README.md) | [Data](../data/README.md)

This directory contains Jupyter notebooks demonstrating programmatic access, multi-modal alignment, digital pathology annotations, and MRI lesion analysis for post-mortem stroke research.

## List of Notebooks

| Aspect / Domain | Notebook | Description | Key Modules / Dependencies |
| :--- | :--- | :--- | :--- |
| **Multi-Modal 3D Visualization** | [`multi_modal_3d_visualization_colab.ipynb`](multi_modal_3d_visualization_colab.ipynb) | Comprehensive Colab/local interactive workflow demonstrating cross-modal alignment between histology ROI stacks, digital pathology annotations, and reference MRI spaces. | `roi_utils`, `annotations_utils`, Matplotlib |
| **Stroke Annotations** | [`stroke_annotations.ipynb`](stroke_annotations.ipynb) | Digital pathology annotation querying, multi-protein GeoJSON retrieval (CD68, GFAP, Fib, CD34, HIF1a), ROI coordinate transforms, filtering, and tissue overlap validation. | `annotations_utils`, OpenAtlas API |
| **Stroke MRI Analysis** | [`stroke_mri.ipynb`](stroke_mri.ipynb) | Core neuroimaging exploration of stroke lesion markers, DWI hyperintensity, ADC hypointensity, and intensity distributions. | `roi_utils`, `annotations_utils`, `mri_utils` |
| **MRI Testing & Validation** | [`MRI_testing.ipynb`](MRI_testing.ipynb) | Comprehensive verification suite for NIfTI geometry, voxel-to-world transforms, brain masking, paired diffusion analysis (iv11 vs iv16), and atlas burden lookups. | `mri_utils`, `nibabel` |
| **MRI Library Exploration** | [`lib-mri.ipynb`](lib-mri.ipynb) | Comparative evaluation using NiBabel and Nilearn for spatial resampling, registration to ex-vivo space, and crosshair mapping. | `nibabel`, `nilearn`, `mri_utils` |
| **Slide Catalogue & Images** | [`slide_catalog_and_images.ipynb`](slide_catalog_and_images.ipynb) | Reference 1 of 4. Blocks, stains, markers and sets; finding slides and their neighbours; pyramid levels, sections, regions and fields; one location across the four stains; block-face photographs; tiles; download table. | `slide_utils`, `display_utils` |
| **Pathology Annotations** | [`pathology_annotations.ipynb`](pathology_annotations.ipynb) | Reference 2 of 4. Annotation classes and grades; polygon measurements, filtering and merging; label masks and labelled tiles; the H&E infarct as spatial reference; markers on neighbouring sections; DAB versus annotation; GeoJSON export. | `pathology_utils`, `slide_utils` |
| **Lesion Across Sections** | [`lesion_across_sections.ipynb`](lesion_across_sections.ipynb) | Reference 3 of 4. Infarct area along the blocks and infarct volume; infarct footprint and 3D surface; marker burden along the blocks; marker profiles by distance from the infarct border. | `pathology_utils`, `slide_utils` |
| **MRI Reference Space** | [`mri_reference_space.ipynb`](mri_reference_space.ipynb) | Reference 4 of 4. In-vivo and ex-vivo series, lesion masks and their agreement, anatomical and arterial attribution, from an MRI voxel to the histology section through it, histology versus MRI lesion section by section. | `refspace_utils`, `slide_utils` |

The four reference notebooks read all data from the C-Stroke platform (`https://c-stroke.humanbrain.in`) and run on Colab, Kaggle or locally without credentials. They are meant to be read in order: images, annotations, the lesion across sections, and MRI.

## Execution Guidelines

1. **Path Resolution**: The notebooks automatically add the repository root to `sys.path` (`sys.path.append('..')`) to import `roi_utils`, `annotations_utils`, and `mri_utils`.
2. **Data Directory**: Imaging volumes and metadata are organized in [`../data/`](../data/README.md). Functions in `mri_utils` automatically resolve data paths across `.` and `data/`.
3. **Environment**: Ensure dependencies are installed via `pip install -r requirements.txt`.
