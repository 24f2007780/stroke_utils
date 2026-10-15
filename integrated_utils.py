"""integrated_utils.py - Modular integration helpers bridging Stroke MRI, Histology, and Annotations.

Composes existing functionality from:
    - mri_utils.py        (MRI loading, masking, coordinate mapping, regional burden)
    - annotations_utils.py (Stroke OpenAtlas loading, coordinate transforms, morphology, density)
    - roi_utils.py        (ROI stack loading, slicing, section resolution)

Key Design Principles:
    1. Modular & Composable: Small, single-purpose functions that can be called independently.
    2. No Invented Registration: Connects MRI and histology at the block/section level where the
       manifest provides mathematically validated mapping; never fabricates in-plane voxel-to-pixel
       alignment that the dataset does not support.
    3. Researcher-Friendly: Clean DataFrame outputs with sensible column names ready for display,
       plotting, and statistical reporting.
    4. Defensive: Gracefully handles missing sections, empty masks, and unannotated regions.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Compose existing utilities
import mri_utils as mu
import annotations_utils as au
import roi_utils as ru

__all__ = [
    "find_lesion_histology_sections",
    "summarize_section_annotations",
    "compare_lesion_annotation_density",
    "summarize_mri_pathology",
    "summarize_lesion_regions",
    "compare_lesion_across_mri",
]


def find_lesion_histology_sections(
    mask_or_points: Union[np.ndarray, pd.DataFrame],
    manifest: dict,
    min_voxels: int = 1,
) -> pd.DataFrame:
    """Identify histology blocks and section ranges intersected by a 3D MRI lesion finding.

    This serves as the primary bridge from a 3D MRI finding (e.g. acute ischemic core mask)
    to physical histology cutting planes, using the oblique slice transformations (krow)
    validated in manifest.json.

    Parameters
    ----------
    mask_or_points : np.ndarray or pd.DataFrame
        Either a 3D boolean lesion mask (aligned to manifest dimensions), or the raw points
        DataFrame output from `mri_utils.map_voxel_to_histology_section`.
    manifest : dict
        Loaded manifest dict (from `mri_utils.load_manifest`).
    min_voxels : int, default=1
        Minimum lesion voxels required in a block to include it in the summary table.

    Returns
    -------
    pd.DataFrame
        Ranked table of histology blocks sorted by descending lesion voxel count:
        Columns:
            - block_id: biosample ID (e.g. 313, 315, 217, 314, 218)
            - block_n: block sequence index (1 to 5)
            - lesion_voxels: number of lesion voxels mapped to this block
            - voxel_fraction: fraction of total mapped lesion voxels in this block
            - section_min, section_max: lowest and highest histology section numbers
            - section_range: human-readable range string (e.g. "24–1842")
            - n_sections: count of unique histology sections spanned
            - global_id_min, global_id_max: global section series range
            - global_id_range: formatted global ID range string
            - mm_min, mm_max, mm_span: physical span along the cutting axis (mm)
    """
    if isinstance(mask_or_points, pd.DataFrame):
        pts_df = mask_or_points
    else:
        pts_df = mu.map_voxel_to_histology_section(mask_or_points, manifest)

    cols = [
        "block_id", "block_n", "lesion_voxels", "voxel_fraction",
        "section_min", "section_max", "section_range", "n_sections",
        "global_id_min", "global_id_max", "global_id_range",
        "mm_min", "mm_max", "mm_span",
    ]

    if pts_df.empty:
        return pd.DataFrame(columns=cols)

    # Focus on voxels within the block's physical slice boundary
    valid = pts_df[pts_df["in_range"]]
    if valid.empty:
        valid = pts_df

    total_valid = len(valid)
    blocks_meta = manifest.get("blocks", {})

    rows = []
    for block_id, group in valid.groupby("block_id"):
        vox_count = len(group)
        if vox_count < min_voxels:
            continue

        b_key = str(block_id)
        b_info = blocks_meta.get(b_key, {})
        block_n = b_info.get("block_n", -1)

        sec_min = int(group["section"].min())
        sec_max = int(group["section"].max())
        n_unique_secs = int(group["section"].nunique())

        gid_min = int(group["global_section_id"].min())
        gid_max = int(group["global_section_id"].max())

        mm_min = round(float(group["mm"].min()), 2)
        mm_max = round(float(group["mm"].max()), 2)
        mm_span = round(abs(mm_max - mm_min), 2)

        rows.append({
            "block_id": int(block_id),
            "block_n": block_n,
            "lesion_voxels": vox_count,
            "voxel_fraction": round(vox_count / total_valid, 4) if total_valid > 0 else 0.0,
            "section_min": sec_min,
            "section_max": sec_max,
            "section_range": f"{sec_min}–{sec_max}",
            "n_sections": n_unique_secs,
            "global_id_min": gid_min,
            "global_id_max": gid_max,
            "global_id_range": f"{gid_min}–{gid_max}",
            "mm_min": mm_min,
            "mm_max": mm_max,
            "mm_span": mm_span,
        })

    if not rows:
        return pd.DataFrame(columns=cols)

    res_df = pd.DataFrame(rows, columns=cols)
    return res_df.sort_values("lesion_voxels", ascending=False, ignore_index=True)


def summarize_section_annotations(
    sections_or_annotations: Union[Iterable[int], au.RoiAnnotations, pd.DataFrame],
    biosample_id: Optional[int] = None,
    stain: str = "IHCS",
    proteins: Optional[List[str]] = None,
    max_sections_to_probe: int = 15,
) -> pd.DataFrame:
    """Report available histology annotations and protein markers for candidate sections.

    Supports two workflows:
      1. Inspection from already loaded RoiAnnotations (instant, offline).
      2. Discovery via OpenAtlas API probes for a given list of section numbers.

    Parameters
    ----------
    sections_or_annotations : Iterable[int], RoiAnnotations, or pd.DataFrame
        - RoiAnnotations: Extracts loaded annotations across stack sections.
        - List/Iterable of ints: Probes availability via find_available_annotations.
        - DataFrame: Takes sections from find_lesion_histology_sections summary.
    biosample_id : int, optional
        Biosample ID (Stroke or Atlas ID). Required if probing explicit section numbers.
    stain : str, default="IHCS"
        Stain code in Stroke OpenAtlas (pathology detections are typically under IHCS).
    proteins : list of str, optional
        Subset of STROKE_PROTEINS to check (default: all candidate proteins).
    max_sections_to_probe : int, default=15
        Safety cap on network probes when checking lists of section numbers.

    Returns
    -------
    pd.DataFrame
        Summary table detailing sections, available proteins, and polygon counts.
    """
    # Case 1: Already loaded RoiAnnotations object
    if isinstance(sections_or_annotations, au.RoiAnnotations) or isinstance(sections_or_annotations, dict):
        roi_ann = sections_or_annotations
        atlas_map = getattr(roi_ann, "atlas_sections", {}) or {}
        rows = []
        for s_idx in sorted(roi_ann.keys()):
            p_dict = roi_ann.get(s_idx, {})
            active_proteins = [p for p, recs in p_dict.items() if recs and (proteins is None or p in proteins)]
            if active_proteins:
                poly_count = sum(len(p_dict[p]) for p in active_proteins)
                atlas_sec = atlas_map.get(s_idx)
                rows.append({
                    "stack_index": s_idx,
                    "atlas_section": atlas_sec,
                    "proteins": ", ".join(active_proteins),
                    "n_polygons": poly_count,
                    "has_annotations": True,
                })
        cols = ["stack_index", "atlas_section", "proteins", "n_polygons", "has_annotations"]
        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows, columns=cols)

    # Case 2: DataFrame from find_lesion_histology_sections
    if isinstance(sections_or_annotations, pd.DataFrame):
        if sections_or_annotations.empty:
            return pd.DataFrame(columns=["section", "biosample_id", "stain", "proteins", "has_annotations"])
        top_row = sections_or_annotations.iloc[0]
        if biosample_id is None:
            biosample_id = int(top_row["block_id"])
        # Select representative endpoints and midpoint to probe
        s_min = int(top_row["section_min"])
        s_max = int(top_row["section_max"])
        sec_list = sorted({s_min, (s_min + s_max) // 2, s_max})
    else:
        sec_list = list(sections_or_annotations)[:max_sections_to_probe]

    if biosample_id is None:
        raise ValueError("biosample_id must be specified when probing section numbers.")

    rows = []
    for sec in sec_list:
        try:
            avail = au.find_available_annotations(
                biosample_id=biosample_id,
                stain=stain,
                section_num=sec,
                proteins=proteins,
                verbose=False,
            )
            found = [a["protein"] for a in avail if a["available"]]
        except Exception:
            found = []

        rows.append({
            "section": sec,
            "biosample_id": biosample_id,
            "stain": stain,
            "proteins": ", ".join(found) if found else "None",
            "n_proteins": len(found),
            "has_annotations": len(found) > 0,
        })

    cols = ["section", "biosample_id", "stain", "proteins", "n_proteins", "has_annotations"]
    return pd.DataFrame(rows, columns=cols)


def compare_lesion_annotation_density(
    roi_annotations: au.RoiAnnotations,
    lesion_section_range: Tuple[int, int],
    proteins: Optional[List[str]] = None,
    units: str = "roi_px",
    mpp: Optional[float] = None,
    return_details: bool = False,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Compare annotation pathology burden inside versus outside an MRI lesion's section range.

    Computes section-level association (total polygon area and detection frequency)
    between histology sections falling within the MRI lesion range and surrounding sections.

    NOTE: This is a section-level correspondence along the cutting axis (Z), NOT an in-plane
    voxel-level spatial registration.

    Parameters
    ----------
    roi_annotations : RoiAnnotations
        Loaded ROI stack annotations (from annotations_utils.load_roi_annotations).
    lesion_section_range : tuple of (int, int)
        (min_section, max_section) defining the Atlas section numbers intersected by the lesion.
    proteins : list of str, optional
        Subset of candidate proteins to evaluate.
    units : str, default="roi_px"
        Measurement units ('roi_px', 'um', or 'mm').
    mpp : float, optional
        Microns per pixel at this ROI level for physical unit conversion.
    return_details : bool, default=False
        If True, returns a dict with {"summary": summary_df, "details": per_section_df}.

    Returns
    -------
    pd.DataFrame or dict
        Summary comparison table per protein:
        Columns:
            - protein: marker name (e.g. CD68, GFAP, FIB)
            - inside_mean_area: mean total area on sections within lesion range
            - outside_mean_area: mean total area on sections outside lesion range
            - inside_annotated_sections: number of annotated slices inside
            - outside_annotated_sections: number of annotated slices outside
            - density_ratio: inside / outside ratio (enrichment metric)
            - association: qualitative summary of section-level correspondence
            - analysis_type: explicitly flags section-level association
    """
    density_rows = au.annotation_density_by_section(
        roi_annotations, proteins=proteins, units=units, mpp=mpp
    )

    summary_cols = [
        "protein", "inside_mean_area", "outside_mean_area",
        "inside_annotated_sections", "outside_annotated_sections",
        "density_ratio", "association", "analysis_type",
    ]

    if not density_rows:
        empty_summary = pd.DataFrame(columns=summary_cols)
        empty_details = pd.DataFrame(columns=["stack_index", "section", "protein", "total_area", "n_annotations", "in_lesion_range"])
        if return_details:
            return {"summary": empty_summary, "details": empty_details}
        return empty_summary

    df = pd.DataFrame(density_rows)
    lo, hi = min(lesion_section_range), max(lesion_section_range)
    df["in_lesion_range"] = df["section"].between(lo, hi)

    summary_records = []
    for protein in sorted(df["protein"].unique()):
        p_df = df[df["protein"] == protein]
        inside = p_df[p_df["in_lesion_range"]]
        outside = p_df[~p_df["in_lesion_range"]]

        in_mean = float(inside["total_area"].mean()) if len(inside) > 0 else 0.0
        out_mean = float(outside["total_area"].mean()) if len(outside) > 0 else 0.0

        in_count = len(inside)
        out_count = len(outside)

        if out_mean > 0:
            ratio = round(in_mean / out_mean, 2)
            if ratio > 1.25:
                assoc = f"Enriched in lesion range ({ratio:.2f}x)"
            elif ratio < 0.8:
                assoc = f"Reduced in lesion range ({ratio:.2f}x)"
            else:
                assoc = f"Comparable density ({ratio:.2f}x)"
        elif in_mean > 0:
            ratio = float("inf")
            assoc = "Present only in lesion range"
        else:
            ratio = 1.0
            assoc = "No detections"

        summary_records.append({
            "protein": protein,
            "inside_mean_area": round(in_mean, 2),
            "outside_mean_area": round(out_mean, 2),
            "inside_annotated_sections": in_count,
            "outside_annotated_sections": out_count,
            "density_ratio": ratio,
            "association": assoc,
            "analysis_type": "Section-level association (cutting axis)",
        })

    summary_df = pd.DataFrame(summary_records, columns=summary_cols)
    if return_details:
        return {"summary": summary_df, "details": df}
    return summary_df


def summarize_mri_pathology(
    lesion_mask: np.ndarray,
    manifest: dict,
    roi_stack: Optional[np.ndarray] = None,
    roi_meta: Optional[dict] = None,
    block_id: Optional[int] = None,
    stain: str = "IHCS",
    proteins: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Orchestrate the complete MRI lesion to histology section and annotation pipeline.

    Workflow:
      1. Maps MRI lesion mask to candidate histology blocks and section intervals.
      2. Selects the primary intersecting block (or user-specified block).
      3. If an ROI stack is provided, loads available pathology annotations and compares
         section-level density inside vs. outside the lesion cutting range.

    Parameters
    ----------
    lesion_mask : np.ndarray
        3D boolean MRI lesion mask (e.g. from lesion_mask or restricted_diffusion_lesion).
    manifest : dict
        Manifest configuration dict.
    roi_stack : np.ndarray, optional
        Loaded ROI NumPy array.
    roi_meta : dict, optional
        ROI metadata dictionary from load_and_preview_roi.
    block_id : int, optional
        Target biosample block ID. If None, automatically selects the primary block
        with highest lesion voxel burden.
    stain : str, default="IHCS"
        Stain code for OpenAtlas annotation retrieval.
    proteins : list of str, optional
        Subset of proteins to query.

    Returns
    -------
    dict
        Structured analysis bundle containing:
            - 'histology_blocks': Summary DataFrame of all intersected blocks
            - 'primary_block_id': Target block ID analyzed
            - 'primary_section_range': (min_sec, max_sec) interval
            - 'roi_annotations': RoiAnnotations instance (if ROI provided)
            - 'annotation_summary': Section annotation summary table
            - 'density_comparison': Inside vs. outside density comparison table
    """
    blocks_df = find_lesion_histology_sections(lesion_mask, manifest)

    if blocks_df.empty:
        return {
            "histology_blocks": blocks_df,
            "primary_block_id": None,
            "primary_section_range": None,
            "roi_annotations": None,
            "annotation_summary": None,
            "density_comparison": None,
            "message": "Lesion mask does not intersect valid histology block coordinates.",
        }

    # Select target block
    if block_id is not None:
        target_block = int(block_id)
        matching = blocks_df[blocks_df["block_id"] == target_block]
        if not matching.empty:
            target_range = (int(matching.iloc[0]["section_min"]), int(matching.iloc[0]["section_max"]))
        else:
            target_range = (1, 2000)
    else:
        top_row = blocks_df.iloc[0]
        target_block = int(top_row["block_id"])
        target_range = (int(top_row["section_min"]), int(top_row["section_max"]))

    roi_annotations = None
    annot_summary = None
    density_comp = None

    if roi_stack is not None and roi_meta is not None:
        try:
            # Query Stroke OpenAtlas under stain (typically IHCS for pathology detections)
            roi_annotations = au.load_roi_annotations(
                roi_stack, roi_meta,
                biosample_id=target_block,
                stain=stain,
                proteins=proteins,
                verbose=False,
            )
            annot_summary = summarize_section_annotations(roi_annotations, proteins=proteins)
            density_comp = compare_lesion_annotation_density(
                roi_annotations, target_range, proteins=proteins
            )
        except Exception as e:
            annot_summary = pd.DataFrame()
            density_comp = pd.DataFrame()

    return {
        "histology_blocks": blocks_df,
        "primary_block_id": target_block,
        "primary_section_range": target_range,
        "roi_annotations": roi_annotations,
        "annotation_summary": annot_summary,
        "density_comparison": density_comp,
    }


def summarize_lesion_regions(
    lesion_mask: np.ndarray,
    manifest: dict,
    atlas: str = "both",
    lr_swap: bool = False,
    top_n: Optional[int] = 10,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Tabulate anatomical and arterial territory infiltration by an MRI lesion.

    Composes `mri_utils.load_manifest_volume` and `mri_utils.regional_lesion_burden`
    to report clinical region breakdown in mL and percentage infiltration.

    Parameters
    ----------
    lesion_mask : np.ndarray
        3D boolean lesion mask.
    manifest : dict
        Loaded manifest dict containing 'atlas' lookup tables.
    atlas : {'both', 'mni', 'art'}, default='both'
        Which atlas to summarize: 'mni' (anatomical), 'art' (arterial), or 'both'.
    lr_swap : bool, default=False
        Whether to swap laterality labels for arterial atlas (see manifest notes).
    top_n : int, optional, default=10
        Limit to top N affected regions per atlas (or None for all).

    Returns
    -------
    pd.DataFrame or dict
        If atlas='both': dict with {'anatomical': df_mni, 'arterial': df_art}.
        Otherwise single DataFrame ranked by lesion_volume_mL.
    """
    results = {}

    if atlas in ("mni", "both"):
        v_mni = mu.load_manifest_volume(manifest, "mni")
        df_mni = mu.regional_lesion_burden(
            lesion_mask, v_mni, manifest["atlas"]["mni"], manifest
        )
        if top_n is not None and len(df_mni) > top_n:
            df_mni = df_mni.head(top_n).copy()
        results["anatomical"] = df_mni

    if atlas in ("art", "both"):
        v_art = mu.load_manifest_volume(manifest, "art")
        df_art = mu.regional_lesion_burden(
            lesion_mask, v_art, manifest["atlas"]["art"], manifest, lr_swap=lr_swap
        )
        if top_n is not None and len(df_art) > top_n:
            df_art = df_art.head(top_n).copy()
        results["arterial"] = df_art

    if atlas == "both":
        return results
    return results.get(atlas, pd.DataFrame())


def compare_lesion_across_mri(
    mri_a: mu.MRI,
    mri_b: mu.MRI,
    mask_a: Optional[np.ndarray] = None,
    mask_b: Optional[np.ndarray] = None,
    label_a: str = "Scan A",
    label_b: str = "Scan B",
) -> pd.DataFrame:
    """Compare lesion volumes, intensity changes, and spatial overlap across two aligned MRI scans.

    Composes `mri_utils.compare_mri_volumes` into a tidy, single-row summary table
    suitable for longitudinal tracking (e.g. 16 dpm vs. 11 dpm evolution).

    Parameters
    ----------
    mri_a, mri_b : MRI
        Loaded MRI objects on shared physical grids (verified via check_spatial_alignment).
    mask_a, mask_b : np.ndarray, optional
        Boolean lesion masks delineated on mri_a and mri_b respectively.
    label_a, label_b : str, default="Scan A", "Scan B"
        Labels identifying each scan timepoint or acquisition.

    Returns
    -------
    pd.DataFrame
        Single-row summary containing volume_mL for each scan, absolute and percent volume change,
        Dice overlap, Jaccard index, and in-mask mean intensities.
    """
    comp = mu.compare_mri_volumes(
        mri_a, mri_b, mask_a=mask_a, mask_b=mask_b, label_a=label_a, label_b=label_b
    )

    vol_a = comp.get(f"volume_mL_{label_a}")
    vol_b = comp.get(f"volume_mL_{label_b}")
    vol_change = comp.get("volume_change_mL")

    pct_change = None
    if vol_a is not None and vol_a > 0 and vol_change is not None:
        pct_change = round(100.0 * vol_change / vol_a, 2)

    row = {
        "scan_a": label_a,
        "scan_b": label_b,
        f"volume_mL_{label_a}": vol_a,
        f"volume_mL_{label_b}": vol_b,
        "volume_change_mL": vol_change,
        "pct_volume_change": pct_change,
        "dice_overlap": comp.get("dice"),
        "jaccard_index": comp.get("jaccard"),
        f"mean_intensity_{label_a}": round(comp.get(f"mean_intensity_{label_a}_in_mask_a", 0.0), 2)
        if f"mean_intensity_{label_a}_in_mask_a" in comp else None,
        f"mean_intensity_{label_b}": round(comp.get(f"mean_intensity_{label_b}_in_mask_a", 0.0), 2)
        if f"mean_intensity_{label_b}_in_mask_a" in comp else None,
    }

    return pd.DataFrame([row])
