# Stroke Dataset Files & Data Manifest

Back to [Top](../README.md) | [Documentation](../docs/README.md) | [Notebooks](../notebooks/README.md)

This directory contains the imaging volumes, atlas label volumes, and coordinate metadata for post-mortem stroke case **HB02CV**.

> [!NOTE]
> Large medical imaging files (`*.nii`, `*.nii.gz`, `*.bin`, `*.bin.gz`) are excluded from Git version control via `.gitignore` to keep the repository lightweight. Metadata files (`manifest.json`, `sets_data.json`) are version-controlled.

---

## File Inventory

### 1. Metadata Files (Tracked)

| File | Size | Description |
| :--- | :--- | :--- |
| `manifest.json` | ~59 KB | Primary reference specification: 3D grid dimensions (`[211, 377, 238]`), voxel spacing (`[0.488, 0.488, 0.70]` mm), axis conventions, histology block plane affine equations (`krow`), MNI & arterial atlas label tables. |
| `sets_data.json` | ~918 KB | Mapping between histology section numbers, block IDs, and stained slide sets (HEOS, NISL, MYEL, IHCS, IHC3) at 20 µm intervals. |

### 2. NIfTI MRI Volumes (Git-Ignored)

| Volume | Format | Description |
| :--- | :--- | :--- |
| `DWI-11.nii` | NIfTI (LPS) | Clinical scanner acquisition (oblique, native scanner frame). |
| `iv11_DWI.nii.gz` | NIfTI (PSR) | In-vivo high-resolution DWI at 11 days pre-mortem. |
| `iv11_ADC.nii.gz` | NIfTI (PSR) | In-vivo high-resolution ADC at 11 days pre-mortem. |
| `iv11_dwi_hyper.nii.gz` | NIfTI (PSR) | Ground-truth / expert segmented DWI hyperintense lesion core mask. |
| `iv16_DWI.nii.gz` | NIfTI (PSR) | In-vivo high-resolution DWI at 16 days pre-mortem. |
| `11dpm_ADC_regto_exvivo_cropped.nii.gz` | NIfTI | 11 dpm ADC co-registered to ex-vivo reference space. |
| `16dpm_DWI_regto_exvivo_cropped.nii.gz` | NIfTI | 16 dpm DWI co-registered to ex-vivo reference space. |
| `16dpm_ADC_regto_exvivo_cropped.nii.gz` | NIfTI | 16 dpm ADC co-registered to ex-vivo reference space. |

### 3. Binary Label & Atlas Volumes (Git-Ignored)

| File | Dimensions | Description |
| :--- | :--- | :--- |
| `lab_mni.bin` | 211 × 377 × 238 | 125 MNI structural anatomical region IDs pre-aligned to ex-vivo reference space. |
| `lab_art.bin` | 211 × 377 × 238 | 32 JHU arterial territory label IDs pre-aligned to ex-vivo reference space. |
| `segs.bin` | 211 × 377 × 238 | Bit-packed binary masks for 7 segmented lesion compartments (iv11, iv16, ex-vivo). |
| `iv11_DWI.bin`, `iv11_ADC.bin` | 211 × 377 × 238 | Grayscale uint8 windowed display volumes for the interactive viewer. |
| `iv16_DWI.bin`, `iv16_ADC.bin` | 211 × 377 × 238 | Grayscale uint8 windowed display volumes for the interactive viewer. |

---

## Data Loading Helpers

Use `mri_utils.py` to seamlessly load data files from any working directory:

```python
import mri_utils as mu

# Load manifest and volumes
manifest = mu.load_manifest("manifest.json")
mri = mu.load_mri("iv11_DWI.nii.gz")
mni_vol = mu.load_manifest_volume(manifest, "mni")
```
