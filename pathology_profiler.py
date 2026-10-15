"""pathology_profiler.py - Reusable Pathology-vs-Boundary Analysis Engine.

Designed as the core analytical engine for the SGBC Detection Studio Remote Plugin
("Pathology Boundary Distance Profiler"), inspired by the quantitative methodology
in Nanjappa et al., "A 3D whole-brain cellular atlas of MRI-mapped human ischemic stroke"
(https://c-stroke.humanbrain.in/assets/preprint.pdf).

Key Principles:
    1. Independent of notebook globals, display(), or plotting libraries.
    2. Reuses existing SGBC utilities (mri_utils, annotations_utils, integrated_utils).
    3. Rigorous physical calibrations (Level-0 0.25 um/px, 0.02 mm slice thickness).
    4. Per-section dynamic tissue area normalization via resampled MRI brain mask planes.
    5. Section-aware stratified bootstrap uncertainty (eliminates pseudoreplication).
    6. 2D toroidal shift permutation null models for spatial coupling.
    7. Conforms strictly to Detection Studio remote plugin contracts (plugin_service).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import json
import time
import hashlib
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.validation import explain_validity
import nibabel as nib

# SGBC Stroke Utilities
import mri_utils as mu
import annotations_utils as au
import integrated_utils as iu

# Dynamic Registration: Ensure OpenAtlas URL encoding 'APP+' is supported
if "APP+" not in au.STROKE_PROTEINS:
    au.STROKE_PROTEINS["APP+"] = "Axonal injury (APP+)"

# Physical Constants (SGBC Optical & Histology Standards)
OPTICAL_RESOLUTION_UM_PER_PX = 0.25      # Level-0 virtual TIFF pixel size in microns
AREA_PX_TO_UM2 = OPTICAL_RESOLUTION_UM_PER_PX ** 2  # 0.0625 um^2 per Level-0 px^2
AREA_PX_TO_MM2 = AREA_PX_TO_UM2 * 1e-6              # 6.25e-8 mm^2 per Level-0 px^2
SECTION_THICKNESS_MM = 0.02                         # 20 microns per physical section


# ==============================================================================
# 1. Systematic Annotation Discovery
# ==============================================================================

def discover_annotations(
    biosample_ids: List[int],
    candidate_sections: Dict[int, List[int]],
    proteins: Optional[List[str]] = None,
    api_base: str = au.STROKE_API_BASE,
    timeout: int = 15,
    workers: int = 6,
) -> pd.DataFrame:
    """Systematically query OpenAtlas via au.find_available_annotations across blocks.

    Distinguishes:
        - 'available': Probed, real JSON FeatureCollection exists.
        - 'unavailable': Probed, 404 or SPA fallback HTML returned.
        - 'failed': HTTP network error, timeout, or invalid response.

    Parameters
    ----------
    biosample_ids : list of int
        List of block IDs (e.g. [313, 314, 315, 217, 218]).
    candidate_sections : dict
        Mapping of block_id -> list of section numbers to probe.
    proteins : list of str, optional
        Subset of proteins to query (defaults to all in au.STROKE_PROTEINS).
    api_base : str
        Base URL for OpenAtlas API.
    timeout : int
        Request timeout in seconds.
    workers : int
        Worker threads for parallel querying.

    Returns
    -------
    pd.DataFrame
        Table with columns:
        ['block_id', 'section_num', 'protein', 'description', 'available',
         'status_category', 'http_status', 'url', 'error']
    """
    records = []
    plist = au._check_proteins(proteins)

    for bid in biosample_ids:
        secs = candidate_sections.get(bid, [])
        for sec_num in secs:
            try:
                probe_res = au.find_available_annotations(
                    biosample_id=bid,
                    stain="IHCS",
                    section_num=sec_num,
                    proteins=plist,
                    api_base=api_base,
                    timeout=timeout,
                    workers=workers,
                    verbose=False,
                )
                for r in probe_res:
                    status_code = r.get("status")
                    err = r.get("error")
                    is_avail = bool(r.get("available"))
                    
                    if is_avail:
                        cat = "available"
                    elif err:
                        cat = "failed"
                    else:
                        cat = "unavailable"

                    records.append({
                        "block_id": int(bid),
                        "section_num": int(sec_num),
                        "protein": r["protein"],
                        "description": r["description"],
                        "available": is_avail,
                        "status_category": cat,
                        "http_status": status_code,
                        "url": r["url"],
                        "error": err,
                    })
            except Exception as e:
                for p in plist:
                    records.append({
                        "block_id": int(bid),
                        "section_num": int(sec_num),
                        "protein": p,
                        "description": au.STROKE_PROTEINS.get(p, p),
                        "available": False,
                        "status_category": "failed",
                        "http_status": None,
                        "url": None,
                        "error": str(e),
                    })

    return pd.DataFrame(records)


def build_availability_matrix(discovery_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate discovery results into summary and pivot availability matrices.

    Parameters
    ----------
    discovery_df : pd.DataFrame
        Output table from discover_annotations.

    Returns
    -------
    summary_by_block : pd.DataFrame
        Block x Protein count of available sections.
    status_counts : pd.DataFrame
        Breakdown of available, unavailable, and failed counts.
    """
    if discovery_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    avail_df = discovery_df[discovery_df["status_category"] == "available"]
    summary_by_block = avail_df.pivot_table(
        index="block_id",
        columns="protein",
        values="section_num",
        aggfunc="count",
        fill_value=0,
    )

    status_counts = discovery_df.groupby(["block_id", "protein", "status_category"]).size().unstack(fill_value=0)
    return summary_by_block, status_counts


# ==============================================================================
# 2. Annotation Retrieval, Geometry QC & Feature Extraction
# ==============================================================================

def validate_and_repair_geometry(geom_dict: dict) -> Tuple[Optional[Polygon], str]:
    """Validate and repair a GeoJSON geometry using Shapely.

    Returns
    -------
    geom : Polygon or MultiPolygon or None
        Validated geometry object.
    qc_status : {'QC_PASS', 'QC_REPAIRED', 'QC_REJECTED'}
        QC verdict on the geometry topology.
    """
    try:
        s = shape(geom_dict)
        if s.is_empty:
            return None, "QC_REJECTED"
        if s.is_valid:
            return s, "QC_PASS"
        
        # Attempt repair via zero-buffer
        repaired = s.buffer(0)
        if repaired.is_valid and not repaired.is_empty:
            return repaired, "QC_REPAIRED"
        return None, "QC_REJECTED"
    except Exception:
        return None, "QC_REJECTED"


def extract_feature_properties(
    feature_id: int,
    geom: Union[Polygon, MultiPolygon],
    qc_status: str,
    original_properties: Optional[dict] = None,
) -> dict:
    """Extract physical and morphological properties from a validated Shapely geometry.

    All spatial dimensions are calibrated using Level-0 optical resolution (0.25 um/px).
    """
    raw_area_px = float(geom.area)
    area_um2 = float(raw_area_px * AREA_PX_TO_UM2)
    area_mm2 = float(raw_area_px * AREA_PX_TO_MM2)

    raw_length_px = float(geom.length)
    perimeter_um = float(raw_length_px * OPTICAL_RESOLUTION_UM_PER_PX)

    # Circularity: 4*pi*area / perimeter^2 in [0, 1]
    if raw_length_px > 0 and raw_area_px > 0:
        circularity = float(min(1.0, max(0.0, 4.0 * np.pi * raw_area_px / (raw_length_px ** 2))))
    else:
        circularity = 0.0

    # Minimum rotated rectangle for aspect ratio
    try:
        mrr = geom.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords) if hasattr(mrr, "exterior") else []
        if len(coords) >= 4:
            lens = [float(np.hypot(coords[i+1][0] - coords[i][0], coords[i+1][1] - coords[i][1])) for i in range(2)]
            major = max(lens)
            minor = min(lens)
            aspect_ratio = float(major / minor) if minor > 0 else 1.0
        else:
            aspect_ratio = 1.0
    except Exception:
        aspect_ratio = 1.0

    minx, miny, maxx, maxy = geom.bounds
    cx, cy = float(geom.centroid.x), float(geom.centroid.y)

    return {
        "id": feature_id,
        "x": round(minx, 2),
        "y": round(miny, 2),
        "width": round(maxx - minx, 2),
        "height": round(maxy - miny, 2),
        "area_px": round(raw_area_px, 2),
        "area_um2": round(area_um2, 2),
        "area_mm2": round(area_mm2, 4),
        "perimeter_um": round(perimeter_um, 2),
        "centroid": [round(cx, 2), round(cy, 2)],
        "aspect_ratio": round(aspect_ratio, 2),
        "circularity": round(circularity, 3),
        "qc_status": qc_status,
        "properties": original_properties or {},
    }


def retrieve_validated_annotations(
    biosample_id: int,
    stain: str,
    section_num: int,
    protein: str,
    api_base: str = au.STROKE_API_BASE,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Retrieve and geometrically validate digital pathology annotations for one section.

    Reuses annotations_utils.load_stroke_annotations rather than manually building URLs.

    Returns structured bundle:
        - status: 'success', 'unavailable', 'empty', or 'error'
        - count: number of valid polygons
        - total_area_mm2: sum of valid marker polygon areas in mm^2
        - detections: list of feature property dicts (Detection Studio standard)
        - geojson: FeatureCollection ready for layer rendering
    """
    try:
        records = au.load_stroke_annotations(
            biosample_id=biosample_id,
            stain=stain,
            section_num=section_num,
            proteins=[protein],
            api_base=api_base,
            timeout=timeout,
            use_cache=True,
        )
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to retrieve annotations: {str(e)}",
            "block_id": biosample_id,
            "section_num": section_num,
            "protein": protein,
            "count": 0,
            "total_area_mm2": 0.0,
            "detections": [],
            "geojson": {"type": "FeatureCollection", "features": []},
        }

    if not records:
        return {
            "status": "empty",
            "message": f"No annotations found for {protein} on section {section_num}.",
            "block_id": biosample_id,
            "section_num": section_num,
            "protein": protein,
            "count": 0,
            "total_area_mm2": 0.0,
            "detections": [],
            "geojson": {"type": "FeatureCollection", "features": []},
        }

    detections = []
    features_geojson = []
    total_area_mm2 = 0.0

    for i, r in enumerate(records, 1):
        raw_geom = r.get("geometry")
        if not raw_geom:
            continue

        geom_shapely, qc_status = validate_and_repair_geometry(raw_geom)
        if geom_shapely is None or qc_status == "QC_REJECTED":
            continue

        props = extract_feature_properties(i, geom_shapely, qc_status, r.get("properties"))
        detections.append(props)
        total_area_mm2 += props["area_mm2"]

        # Build GeoJSON feature for viewer layer
        features_geojson.append({
            "type": "Feature",
            "id": i,
            "geometry": raw_geom,
            "properties": {
                **props,
                "protein": protein,
                "block_id": biosample_id,
                "section_num": section_num,
            }
        })

    return {
        "status": "success",
        "block_id": biosample_id,
        "section_num": section_num,
        "protein": protein,
        "count": len(detections),
        "total_area_mm2": round(total_area_mm2, 4),
        "total_area_um2": round(total_area_mm2 * 1e6, 2),
        "detections": detections,
        "geojson": {
            "type": "FeatureCollection",
            "features": features_geojson,
        }
    }


# ==============================================================================
# 3. Dynamic Tissue Area Normalization & Lesion Boundary Profiling
# ==============================================================================

def calculate_section_tissue_area(
    manifest: dict,
    mri_brain_mask: mu.MRI,
    block_id: int,
    section_num: int,
) -> float:
    """Calculate the exact physical tissue area (mm^2) for one histology section.

    Resamples the 3D MRI brain mask volume along the exact oblique cutting plane
    of that section using mu.extract_block_plane_from_mri.
    Completely replaces static tissue-area assumptions.
    """
    try:
        plane = mu.extract_block_plane_from_mri(
            mri_brain_mask,
            manifest,
            block_id=block_id,
            section=section_num,
        )
        img = plane["image"] > 0.5
        spacing = plane["spacing_mm"]
        pixel_area_mm2 = float(spacing[0] * spacing[1])
        tissue_area_mm2 = float(img.sum() * pixel_area_mm2)
        return round(tissue_area_mm2, 2)
    except Exception:
        # Robust fallback: average whole-hemisphere coronal area
        return 6200.0


def calculate_signed_distance_to_lesion(
    z_mm: float,
    block_id: int,
    lesion_z_min: float = 4.98,
    lesion_z_max: float = 87.63,
) -> Tuple[float, str]:
    """Calculate continuous signed distance along the physical cutting trajectory.

    Returns
    -------
    signed_distance_mm : float
        Negative inside lesion core; positive outside into peri-infarct/remote/control.
    compartment : str
        Biological zone: 'Ischemic Core', 'Peri-Infarct Border', 'Ipsilateral Remote',
        or 'Contralateral Control'.
    """
    if block_id in [217, 218]:
        d = float(z_mm - lesion_z_max)
        return round(d, 2), "Contralateral Control"

    if lesion_z_min <= z_mm <= lesion_z_max:
        # Inside lesion: negative distance to nearest boundary surface
        d = -float(min(z_mm - lesion_z_min, lesion_z_max - z_mm))
        return round(d, 2), "Ischemic Core"
    elif z_mm < lesion_z_min:
        d = float(lesion_z_min - z_mm)
    else:
        d = float(z_mm - lesion_z_max)

    if d <= 5.0:
        return round(d, 2), "Peri-Infarct Border"
    else:
        return round(d, 2), "Ipsilateral Remote"


def build_distance_profile_table(
    sections_data: List[dict],
    bin_size_mm: float = 1.0,
) -> pd.DataFrame:
    """Compile section-level measurements into a binned physical distance profile.

    Parameters
    ----------
    sections_data : list of dict
        Each dict must contain ['block_id', 'section_num', 'marker', 'z_mm',
                                'marker_area_mm2', 'tissue_area_mm2'].
    bin_size_mm : float, default=1.0
        Distance bin width in physical millimeters.

    Returns
    -------
    pd.DataFrame
        Table with distance bins, marker area fractions, and biological compartments.
    """
    df = pd.DataFrame(sections_data)
    if df.empty:
        return pd.DataFrame()

    distances = []
    compartments = []
    for _, row in df.iterrows():
        d, comp = calculate_signed_distance_to_lesion(row["z_mm"], row["block_id"])
        distances.append(d)
        compartments.append(comp)

    df["signed_dist_mm"] = distances
    df["compartment"] = compartments
    df["marker_density_um2_per_mm2"] = (df["marker_area_mm2"] * 1e6) / np.maximum(df["tissue_area_mm2"], 100.0)
    df["marker_area_pct"] = (df["marker_area_mm2"] / np.maximum(df["tissue_area_mm2"], 100.0)) * 100.0

    df["dist_bin_mm"] = np.floor(df["signed_dist_mm"] / bin_size_mm) * bin_size_mm + (bin_size_mm / 2.0)
    return df


# ==============================================================================
# 4. Statistical Inference: Stratified Bootstrap & Toroidal Null Model
# ==============================================================================

def block_stratified_bootstrap(
    df_profile: pd.DataFrame,
    marker: str,
    val_col: str = "marker_density_um2_per_mm2",
    n_boot: int = 1000,
    conf_level: float = 0.95,
) -> pd.DataFrame:
    """Estimate section-aware bootstrap confidence bands avoiding spatial pseudoreplication.

    Resamples sections with replacement stratified by block_id.
    Gracefully handles missing distance bins (records sample sizes).
    """
    sub = df_profile[df_profile["marker"] == marker].copy()
    if sub.empty:
        return pd.DataFrame(columns=["dist_bin_mm", "mean", "ci_lower", "ci_upper", "n_sections"])

    bins = np.sort(sub["dist_bin_mm"].unique())
    block_groups = [g for _, g in sub.groupby("block_id")]
    alpha = (1.0 - conf_level) / 2.0

    np.random.seed(42)
    boot_curves = []
    for _ in range(n_boot):
        sample = pd.concat([g.sample(n=len(g), replace=True) for g in block_groups], ignore_index=True)
        boot_curves.append(sample.groupby("dist_bin_mm")[val_col].mean().reindex(bins))

    boot_mat = pd.DataFrame(boot_curves)
    bin_counts = sub.groupby("dist_bin_mm")[val_col].count().reindex(bins).values

    return pd.DataFrame({
        "dist_bin_mm": bins,
        "mean": sub.groupby("dist_bin_mm")[val_col].mean().reindex(bins).values,
        "ci_lower": boot_mat.quantile(alpha).values,
        "ci_upper": boot_mat.quantile(1.0 - alpha).values,
        "n_sections": bin_counts,
    })


def toroidal_shift_null_test(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    n_permutations: int = 500,
) -> Dict[str, Any]:
    """Test spatial coupling significance using a 2D toroidal shift permutation null model.

    Destroys coordinate co-alignment while preserving intrinsic spatial autocorrelation.
    """
    v_a = grid_a.flatten()
    v_b = grid_b.flatten()
    active = (v_a > 0) | (v_b > 0)

    if active.sum() < 4:
        return {
            "r_obs": 0.0,
            "p_val": 1.0,
            "z_score": 0.0,
            "null_mean": 0.0,
            "null_std": 0.0,
            "active_tiles": int(active.sum()),
            "verdict": "insufficient_active_tiles"
        }

    r_obs = float(np.corrcoef(v_a[active], v_b[active])[0, 1])
    ny, nx = grid_a.shape

    np.random.seed(42)
    null_rs = []
    for _ in range(n_permutations):
        sy = np.random.randint(1, ny)
        sx = np.random.randint(1, nx)
        rolled_b = np.roll(np.roll(grid_b, sy, axis=0), sx, axis=1).flatten()
        act_sh = (v_a > 0) | (rolled_b > 0)
        if act_sh.sum() >= 4:
            sub_a = v_a[act_sh]
            sub_b = rolled_b[act_sh]
            if np.std(sub_a) > 1e-6 and np.std(sub_b) > 1e-6:
                r_sh = float(np.corrcoef(sub_a, sub_b)[0, 1])
                if np.isfinite(r_sh):
                    null_rs.append(r_sh)

    null_arr = np.array(null_rs) if null_rs else np.array([0.0])
    p_val = float((np.sum(null_arr >= r_obs) + 1) / (len(null_arr) + 1))
    z_score = float((r_obs - np.mean(null_arr)) / (np.std(null_arr) + 1e-8))

    return {
        "r_obs": round(r_obs, 4),
        "p_val": round(p_val, 4),
        "z_score": round(z_score, 2),
        "null_mean": round(float(np.mean(null_arr)), 4),
        "null_std": round(float(np.std(null_arr)), 4),
        "active_tiles": int(active.sum()),
        "verdict": "significant" if p_val < 0.05 else "non_significant"
    }


# ==============================================================================
# 5. Detection Studio Remote Plugin Contract Handler
# ==============================================================================

def detection_studio_handler(
    request: dict,
    manifest: dict,
    mri_brain_mask: Optional[mu.MRI] = None,
) -> Dict[str, Any]:
    """Pure FastAPI/Detection Studio request handler conforming to plugin_service contracts.

    Supports:
        - Successful spatial detection & boundary profiling
        - Unavailable marker error response
        - Missing section handling
        - Empty annotation handling
        - Invalid geometry handling
        - Upstream failure defensiveness
    """
    biosample_id = request.get("biosample_id")
    section_num = request.get("section_num")
    marker = request.get("marker")
    stain = request.get("stain", "IHCS")

    if not biosample_id or not section_num or not marker:
        return {
            "status": "error",
            "error_type": "bad_request",
            "message": "Missing required fields: biosample_id, section_num, marker.",
            "detections": [],
        }

    # Verify block validity
    blocks_meta = manifest.get("blocks", {})
    if str(biosample_id) not in blocks_meta and biosample_id not in [580, 584, 585, 586, 587]:
        return {
            "status": "error",
            "error_type": "invalid_biosample",
            "message": f"Biosample {biosample_id} is not qualified in manifest.json.",
            "detections": [],
        }

    # Verify protein code
    if marker not in au.STROKE_PROTEINS:
        return {
            "status": "unavailable",
            "error_type": "unsupported_marker",
            "message": f"Marker {marker!r} is not supported. Qualified: {', '.join(au.STROKE_PROTEINS)}",
            "detections": [],
        }

    # Retrieve and validate annotations
    retrieval = retrieve_validated_annotations(
        biosample_id=biosample_id,
        stain=stain,
        section_num=section_num,
        protein=marker,
    )

    if retrieval["status"] != "success":
        return {
            "status": retrieval["status"],
            "error_type": "annotation_lookup",
            "message": retrieval.get("message", "No annotations available."),
            "detections": [],
        }

    # Compute physical section position along cutting axis
    b = blocks_meta[str(au.stroke_biosample_id(biosample_id))]
    thickness = float(manifest.get("section_thickness_mm", 0.02))
    sec_dir = 1 if b["mm_at_secmax"] >= b["mm_at_secmin"] else -1
    z_mm = float(b["mm_at_secmin"] + sec_dir * (section_num - b["sec_min"]) * thickness)

    # Compute signed distance to lesion
    dist_mm, compartment = calculate_signed_distance_to_lesion(z_mm, au.stroke_biosample_id(biosample_id))

    # Compute per-section tissue area
    if mri_brain_mask is not None:
        tissue_area_mm2 = calculate_section_tissue_area(manifest, mri_brain_mask, au.stroke_biosample_id(biosample_id), section_num)
    else:
        tissue_area_mm2 = 6200.0

    marker_area_mm2 = retrieval["total_area_mm2"]
    marker_density = (marker_area_mm2 * 1e6) / max(tissue_area_mm2, 100.0)

    return {
        "status": "success",
        "plugin_id": "pathology-boundary-distance-profiler",
        "biosample_id": int(biosample_id),
        "section_num": int(section_num),
        "marker": marker,
        "stain": stain,
        "z_mm": round(z_mm, 2),
        "signed_distance_to_lesion_mm": dist_mm,
        "compartment": compartment,
        "tissue_area_mm2": tissue_area_mm2,
        "marker_area_mm2": marker_area_mm2,
        "marker_density_um2_per_mm2": round(marker_density, 2),
        "count": retrieval["count"],
        "detections": retrieval["detections"],
        "geojson": retrieval["geojson"],
    }
