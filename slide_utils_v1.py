"""slide_utils - public access to the C-Stroke whole-slide images, slide catalogue and histology annotations.

Everything here talks only to the public C-Stroke deployment (https://c-stroke.humanbrain.in), so notebooks built
on it run unchanged on Google Colab, Kaggle or any machine with internet access. No Google Drive file, internal
service or database is required.

Data sources
------------
* Slide catalogue   {BASE_URL}/sets_data.json
      505 "sets". Each set anchors one immunohistochemistry (IHC) section and lists its nearest Nissl (N), H&E (HE)
      and myelin (M) sections of the same block (up to four slides), with z position and image path. A routine-stain
      section can be the nearest neighbour of several IHC sections and then belongs to each of those sets.
* Slide images      {BASE_URL}/fcgi-bin/iipsrv.fcgi   (IIPImage server)
      Every slide is a 192000 x 192000 px pyramid (8 levels, 1500 ... 192000 px). Any rectangle can be requested
      at any output width with  ?FIF=<path>&RGN=<x>,<y>,<w>,<h>&WID=<px>&CVT=jpeg  (RGN in fractions of the image).
* Annotations       {BASE_URL}/histology_detections/openAtlasJson/<PROTEIN>/<BIOSAMPLE>/<STAIN>/<SECTION>/...json
      GeoJSON FeatureCollections of graded region polygons (e.g. "sparse / moderately dense / dense astrocytes").

Coordinates
-----------
All geometry in this module is expressed in level-0 slide pixels (x to the right, y down, origin top-left of the
192000 px canvas). Annotation files store (X, Y) with Y negative, so  x = X,  y = -Y.  An image fetched with
``downsample=d`` from origin ``(x0, y0)`` maps pixel (row, col) to level-0 (x0 + col * d, y0 + row * d).

MPP_LEVEL0 (~0.5 um per level-0 pixel) is an estimate derived from section dimensions and nuclear size; the
scanner calibration is not published. Override it if you know the exact value.

Typical use
-----------
    import slide_utils as su
    cat = su.load_slide_catalog()
    slide = su.find_slides(cat, protein="CD68", annotated=True).iloc[0]
    image, region = su.fetch_tissue(slide.path, downsample=8)
    annotations = su.load_annotations(slide.annotation_url)
    grades = su.rasterize_annotations(annotations, region)
"""

import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np

__all__ = [
    "BASE_URL",
    "IIP_URL",
    "SETS_URL",
    "SLIDE_SIZE",
    "MPP_LEVEL0",
    "STAIN_CODES",
    "STAIN_LABELS",
    "PROTEIN_INFO",
    "Region",
    "load_slide_catalog",
    "find_slides",
    "slide_set",
    "fetch_region",
    "fetch_thumbnail",
    "tissue_mask",
    "tissue_bbox",
    "fetch_tissue",
    "tile_features",
    "load_annotations",
    "annotation_classes",
    "rasterize_annotations",
    "stain_separation",
    "tile_reduce",
    "tile_labels",
    "PROTEIN_COLORS",
    "GRADE_COLORS",
    "set_plot_style",
    "draw_annotation_outlines",
    "scale_bar",
]

BASE_URL = os.environ.get("CSTROKE_BASE_URL", "https://c-stroke.humanbrain.in").rstrip("/")
IIP_URL = f"{BASE_URL}/fcgi-bin/iipsrv.fcgi"
SETS_URL = f"{BASE_URL}/sets_data.json"

SLIDE_SIZE = 192000     # level-0 width and height of every slide canvas (px)
MPP_LEVEL0 = 0.5        # estimated microns per level-0 pixel (see module docstring)

# Short stain keys used in sets_data.json -> stain folder on the image server, and readable labels.
STAIN_CODES = {"IHC": "IHCS", "N": "NISL", "HE": "HEOS", "M": "MYEL"}
STAIN_LABELS = {"IHC": "Immunohistochemistry", "N": "Nissl", "HE": "H&E", "M": "Myelin"}

# Canonical protein / annotation keys. `classes` lists annotation class names in increasing severity; the grade
# of a polygon is its 1-based position in that list (0 = not annotated).
PROTEIN_INFO = {
    "CD68":  {"marker": "Microglia / macrophages", "classes": ["moderately dense CD68+ immune cells",
                                                               "dense CD68+ immune cells"]},
    "GFAP":  {"marker": "Reactive astrocytes", "classes": ["sparse astrocytes", "moderately dense astrocytes",
                                                           "dense astrocytes"]},
    "CD34":  {"marker": "Microvessels", "classes": ["sparse microvessels", "moderately dense microvessels",
                                                    "dense microvessels"]},
    "HIF1a": {"marker": "Hypoxia", "classes": ["moderately dense HIF1α+ cells", "dense HIF1α+ cells"]},
    "FIB":   {"marker": "Blood-brain barrier leakage (fibrinogen)", "classes": ["blood brain barrier disruption"]},
    "APP":   {"marker": "Axonal injury", "classes": ["axonal injury"]},
    "GAP43": {"marker": "Axonal regrowth", "classes": ["axonal regrowth"]},
    "AQP4":  {"marker": "Water channel (astrocytic end-feet)", "classes": []},
    "HE":    {"marker": "Infarct (H&E)", "classes": ["Infarcted region"]},
}

# Spellings found in slide file names and annotation folders -> canonical key.
_PROTEIN_ALIASES = {
    "CD68": "CD68", "GFAP": "GFAP", "CD34": "CD34", "AQP4": "AQP4", "GAP43": "GAP43",
    "HIF1A": "HIF1a", "H1F1A": "HIF1a", "HIFIA": "HIF1a",
    "FIB": "FIB", "FIBRINOGEN": "FIB", "FIBRINOGER": "FIB",
    "APP": "APP", "APP+": "APP", "HEOS": "HE",
}

_CACHE_DIR = Path(os.environ.get("CSTROKE_CACHE", Path.home() / ".cache" / "cstroke"))
_SESSION = None


# --------------------------------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------------------------------
def _session():
    global _SESSION
    if _SESSION is None:
        import requests
        from requests.adapters import HTTPAdapter
        _SESSION = requests.Session()
        _SESSION.mount("https://", HTTPAdapter(pool_connections=16, pool_maxsize=16))
        _SESSION.headers["User-Agent"] = "cstroke-notebooks/1.0"
    return _SESSION


def _get(url, timeout=120, retries=3):
    """GET with a short exponential back-off; raises the last error after `retries` attempts."""
    err = None
    for attempt in range(retries):
        try:
            r = _session().get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:  # network errors and 5xx are both worth one more try
            err = e
            time.sleep(1.5 * 2 ** attempt)
    raise RuntimeError(f"Request failed after {retries} attempts: {url}\n{err}")


def _cached_bytes(url, name, use_cache):
    path = _CACHE_DIR / name
    if use_cache and path.exists():
        return path.read_bytes()
    data = _get(url).content
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def _safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


# --------------------------------------------------------------------------------------------------------------
# Slide catalogue
# --------------------------------------------------------------------------------------------------------------
def _canonical_protein(name):
    if not name:
        return None
    return _PROTEIN_ALIASES.get(name.strip().upper(), name.strip())


def load_slide_catalog(url=SETS_URL, use_cache=True):
    """Return the public slide catalogue as a DataFrame with one row per slide and set.

    Columns
        set_id             set number (0-based); the slides of a set are physically adjacent sections
        slot               1 = IHC anchor, 2 = Nissl, 3 = H&E, 4 = myelin (as ordered in sets_data.json)
        biosample          block id (217, 218, 313, 314, 315)
        section            section number within the block
        stain              IHC / N / HE / M
        protein            canonical protein for IHC slides (from the file name), "HE" for H&E, else None
        path               image path on the IIP server
        annotation_url     absolute URL of the annotation file, or None
        annotation_key     canonical key of that annotation (protein, or "HE" for infarct)
        z_mm               section position along the block axis (mm)
        dist_from_anchor_mm  signed distance to the set's IHC section (mm)
        global_id          global section index across blocks
    """
    import pandas as pd

    raw = json.loads(_cached_bytes(url, "sets_data.json", use_cache))
    rows = []
    for entry in raw["sections"]:
        for slot in range(1, 5):
            rec = entry.get(f"section_{slot}")
            if not rec or not rec.get("path"):
                continue
            stain = rec["stain"]
            protein = None
            if stain == "IHC":
                m = re.search(r"\]\[([^\]]+)\]", rec["path"])
                protein = _canonical_protein(m.group(1)) if m else None
            elif stain == "HE":
                protein = "HE"
            ann = rec.get("annotation") or None
            rows.append({
                "set_id": entry["index"],
                "slot": slot,
                "biosample": int(rec["bid"]),
                "section": int(rec["secID"]),
                "stain": stain,
                "protein": protein,
                "path": rec["path"],
                "annotation_url": f"{BASE_URL}/{ann.lstrip('/')}" if ann else None,
                "annotation_key": _canonical_protein(ann.split("/")[2]) if ann else None,
                "z_mm": rec.get("z_mm"),
                "dist_from_anchor_mm": rec.get("dist_from_anchor"),
                "global_id": rec.get("globalID"),
            })
    # A Nissl, H&E or myelin slide is often the nearest neighbour of several IHC anchors and then appears in each
    # of those sets. Rows are kept per set; use .drop_duplicates(["biosample", "stain", "section"]) to count slides.
    return pd.DataFrame(rows)


def find_slides(catalog, protein=None, stain=None, biosample=None, annotated=None):
    """Filter the catalogue. `annotated=True` keeps slides that have an annotation file."""
    sel = catalog
    if protein is not None:
        sel = sel[sel["protein"] == _canonical_protein(protein)]
    if stain is not None:
        sel = sel[sel["stain"] == stain]
    if biosample is not None:
        sel = sel[sel["biosample"].isin(np.atleast_1d(biosample))]
    if annotated is not None:
        sel = sel[sel["annotation_url"].notna() == bool(annotated)]
    return sel.reset_index(drop=True)


def slide_set(catalog, set_id):
    """All slides of one set (IHC anchor plus its Nissl, H&E and myelin neighbours)."""
    return catalog[catalog["set_id"] == set_id].sort_values("slot").reset_index(drop=True)


# --------------------------------------------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Region:
    """A fetched image window: level-0 origin (x0, y0), downsample factor and output shape (rows, cols)."""
    x0: float
    y0: float
    downsample: float
    shape: tuple

    @property
    def mpp(self):
        """Microns per pixel of the fetched image (uses MPP_LEVEL0)."""
        return MPP_LEVEL0 * self.downsample

    @property
    def extent(self):
        """matplotlib imshow extent in level-0 pixels: (left, right, bottom, top)."""
        h, w = self.shape[:2]
        return (self.x0, self.x0 + w * self.downsample, self.y0 + h * self.downsample, self.y0)

    def to_pixels(self, x, y):
        """Level-0 coordinates -> (col, row) in this image."""
        return (np.asarray(x) - self.x0) / self.downsample, (np.asarray(y) - self.y0) / self.downsample

    def to_level0(self, col, row):
        """(col, row) in this image -> level-0 coordinates."""
        return self.x0 + np.asarray(col) * self.downsample, self.y0 + np.asarray(row) * self.downsample


def _decode(data):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def fetch_region(path, x0, y0, width, height, downsample=1.0, use_cache=True):
    """Fetch a level-0 rectangle (x0, y0, width, height) at 1/downsample resolution as uint8 RGB.

    The IIP server resamples from the nearest pyramid level, so any downsample >= 1 is cheap.
    Returns (image, Region).
    """
    out_w = max(1, int(round(width / downsample)))
    out_h = max(1, int(round(height / downsample)))
    rgn = ",".join(f"{v / SLIDE_SIZE:.8f}" for v in (x0, y0, width, height))
    url = f"{IIP_URL}?FIF={quote(path)}&RGN={rgn}&WID={out_w}&CVT=jpeg"
    name = f"tiles/{_safe_name(Path(path).stem)}/{int(x0)}_{int(y0)}_{int(width)}_{int(height)}_{out_w}.jpg"
    try:
        img = _decode(_cached_bytes(url, name, use_cache))
    except OSError:  # the server answered with an error message instead of an image
        (_CACHE_DIR / name).unlink(missing_ok=True)
        raise RuntimeError(f"Image region not available: {path}") from None
    if img.shape[0] != out_h or img.shape[1] != out_w:   # server rounding can differ by one pixel
        from PIL import Image
        img = np.asarray(Image.fromarray(img).resize((out_w, out_h), Image.BILINEAR))
    return img, Region(float(x0), float(y0), float(downsample), img.shape[:2])


def fetch_thumbnail(path, width=2048, use_cache=True):
    """Whole-canvas overview of a slide at `width` px. Returns (image, Region)."""
    return fetch_region(path, 0, 0, SLIDE_SIZE, SLIDE_SIZE, downsample=SLIDE_SIZE / width, use_cache=use_cache)


def tissue_mask(rgb, od_threshold=0.03, sigma=1.0, min_object_frac=1e-4):
    """Boolean tissue mask from optical density (glass is near white, any stain raises OD).

    od_threshold     summed-channel optical density above which a pixel counts as tissue
    sigma            Gaussian smoothing (pixels) applied before thresholding
    min_object_frac  objects smaller than this fraction of the image are discarded
    """
    from scipy import ndimage as ndi

    od = -np.log10(np.clip(rgb.astype(np.float32) + 1.0, 1.0, 256.0) / 256.0).sum(axis=-1)
    if sigma:
        od = ndi.gaussian_filter(od, sigma)
    mask = ndi.binary_fill_holes(od > od_threshold)
    lab, n = ndi.label(mask)
    if n:
        sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
        keep = np.flatnonzero(sizes >= min_object_frac * mask.size) + 1
        mask = np.isin(lab, keep)
    return mask


def tissue_bbox(path, margin=0.01, thumb_width=2048, use_cache=True):
    """Level-0 bounding box (x0, y0, width, height) of the tissue on a slide, from its thumbnail."""
    thumb, reg = fetch_thumbnail(path, width=thumb_width, use_cache=use_cache)
    mask = tissue_mask(thumb)
    if not mask.any():
        raise ValueError(f"No tissue found on {path}")
    rows, cols = np.nonzero(mask)
    pad = margin * SLIDE_SIZE
    x0, x1 = cols.min() * reg.downsample - pad, (cols.max() + 1) * reg.downsample + pad
    y0, y1 = rows.min() * reg.downsample - pad, (rows.max() + 1) * reg.downsample + pad
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(SLIDE_SIZE), x1), min(float(SLIDE_SIZE), y1)
    return x0, y0, x1 - x0, y1 - y0


def fetch_tissue(path, downsample=8, bbox=None, chunk=4096, workers=4, use_cache=True, verbose=True):
    """Fetch the whole tissue area of a slide at 1/downsample resolution as one RGB mosaic.

    The area is requested in chunks of `chunk` output pixels (in parallel) and stitched. With the default
    downsample=8 (~4 um/px) a full hemisphere section is roughly 20000 x 10000 px (~600 MB as uint8 RGB);
    use downsample=16 for a quicker overview.
    Returns (image, Region).
    """
    if bbox is None:
        bbox = tissue_bbox(path, use_cache=use_cache)
    x0, y0, w, h = bbox
    d = float(downsample)
    out_w, out_h = int(round(w / d)), int(round(h / d))
    mosaic = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
    jobs = [(r, c) for r in range(0, out_h, chunk) for c in range(0, out_w, chunk)]

    def _one(job):
        r, c = job
        hh, ww = min(chunk, out_h - r), min(chunk, out_w - c)
        img, _ = fetch_region(path, x0 + c * d, y0 + r * d, ww * d, hh * d, downsample=d, use_cache=use_cache)
        return r, c, img

    t = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r, c, img in ex.map(_one, jobs):
            mosaic[r:r + img.shape[0], c:c + img.shape[1]] = img
    if verbose:
        print(f"Fetched {out_w} x {out_h} px ({MPP_LEVEL0 * d:.1f} um/px, {len(jobs)} requests) "
              f"in {time.time() - t:.1f} s")
    return mosaic, Region(float(x0), float(y0), d, (out_h, out_w))


def tile_features(path, feature_fn, tile=64, downsample=8, bbox=None, chunk_tiles=48, preview_factor=8,
                  workers=4, use_cache=True, verbose=True):
    """Stream a slide chunk by chunk and average per-pixel features over a regular tile grid.

    Only one chunk per worker is held in memory, so whole sections can be processed at cellular resolution.

    feature_fn      rgb chunk (uint8, H x W x 3) -> {name: 2-D float/bool array of the same H x W}
    tile            tile edge in fetched pixels (tile * downsample * MPP_LEVEL0 um on the slide)
    downsample      fetch resolution relative to level 0 (8 -> ~4 um/px)
    preview_factor  an RGB preview is kept at downsample * preview_factor for plotting

    Returns (features, grid, preview, preview_region):
        features        {name: (rows, cols) float32 tile means}
        grid            Region of the tile grid (one pixel per tile)
        preview         uint8 RGB overview of the same area
        preview_region  Region of the preview
    """
    from PIL import Image

    if bbox is None:
        bbox = tissue_bbox(path, use_cache=use_cache)
    x0, y0, w, h = bbox
    d = float(downsample)
    tile_l0 = tile * d
    n_rows, n_cols = int(h // tile_l0), int(w // tile_l0)
    if n_rows == 0 or n_cols == 0:
        raise ValueError("Tile is larger than the tissue area; reduce `tile` or `downsample`.")
    step = chunk_tiles
    jobs = [(r, c) for r in range(0, n_rows, step) for c in range(0, n_cols, step)]
    pf = preview_factor
    tile_px_preview = tile // pf if tile % pf == 0 else None
    if tile_px_preview is None:
        raise ValueError("`tile` must be a multiple of `preview_factor`.")
    preview = np.full((n_rows * tile_px_preview, n_cols * tile_px_preview, 3), 255, dtype=np.uint8)
    features = {}

    def _one(job):
        r, c = job
        nr, nc = min(step, n_rows - r), min(step, n_cols - c)
        img, _ = fetch_region(path, x0 + c * tile_l0, y0 + r * tile_l0, nc * tile_l0, nr * tile_l0,
                              downsample=d, use_cache=use_cache)
        img = img[:nr * tile, :nc * tile]
        feats = {k: tile_reduce(np.asarray(v, dtype=np.float32), tile) for k, v in feature_fn(img).items()}
        small = np.asarray(Image.fromarray(img).resize((nc * tile_px_preview, nr * tile_px_preview), Image.BOX))
        return r, c, feats, small

    t = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r, c, feats, small in ex.map(_one, jobs):
            for k, v in feats.items():
                if k not in features:
                    features[k] = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
                features[k][r:r + v.shape[0], c:c + v.shape[1]] = v
            preview[r * tile_px_preview:r * tile_px_preview + small.shape[0],
                    c * tile_px_preview:c * tile_px_preview + small.shape[1]] = small
    if verbose:
        print(f"{n_rows} x {n_cols} tiles of {tile_l0 * MPP_LEVEL0:.0f} um at {MPP_LEVEL0 * d:.1f} um/px "
              f"({len(jobs)} requests, {time.time() - t:.1f} s)")
    grid = Region(float(x0), float(y0), tile_l0, (n_rows, n_cols))
    preview_region = Region(float(x0), float(y0), d * pf, preview.shape[:2])
    return features, grid, preview, preview_region


# --------------------------------------------------------------------------------------------------------------
# Annotations
# --------------------------------------------------------------------------------------------------------------
def _grade(key, label):
    classes = PROTEIN_INFO.get(key, {}).get("classes", [])
    return classes.index(label) + 1 if label in classes else 1


def load_annotations(url, use_cache=True):
    """Load an annotation file as a list of dicts with keys
    key (protein / "HE"), label (class name), grade (1 = mildest class), geometry (shapely, level-0 px)."""
    from shapely.geometry import shape
    from shapely.ops import transform

    if url is None:
        return []
    key = _canonical_protein(url.split("openAtlasJson/")[1].split("/")[0]) if "openAtlasJson/" in url else None
    name = f"annotations/{_safe_name(url.split('openAtlasJson/')[-1])}"
    try:
        coll = json.loads(_cached_bytes(url, name, use_cache))
    except json.JSONDecodeError:
        # A missing file is answered with the site's HTML landing page rather than a 404.
        (_CACHE_DIR / name).unlink(missing_ok=True)
        raise ValueError(f"Annotation file not available on the server: {url}") from None
    out = []
    for feat in coll.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue
        g = transform(lambda x, y, z=None: (x, -np.asarray(y)), shape(geom))
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty:
            continue
        data = (feat.get("properties") or {}).get("data") or {}
        label = data.get("name") or data.get("text") or "annotation"
        out.append({"key": key, "label": label, "grade": _grade(key, label), "geometry": g})
    return out


def annotation_classes(key):
    """Class names for a protein key in increasing severity, prefixed by 'not annotated' (grade 0)."""
    return ["not annotated"] + PROTEIN_INFO.get(_canonical_protein(key), {}).get("classes", [])


def _polygons(g):
    return list(g.geoms) if hasattr(g, "geoms") else [g]


def rasterize_annotations(annotations, region, value="grade"):
    """Burn annotations into an int array of `region.shape`; overlapping polygons keep the highest value.

    value  "grade" (class severity), "binary" (1 inside any polygon) or a callable rec -> int.
    """
    from PIL import Image, ImageDraw

    h, w = region.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    get = (lambda r: r["grade"]) if value == "grade" else (lambda r: 1) if value == "binary" else value
    for level in sorted({int(get(r)) for r in annotations}):
        canvas = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(canvas)
        for rec in annotations:
            if int(get(rec)) != level:
                continue
            for poly in _polygons(rec["geometry"]):
                if poly.geom_type != "Polygon":
                    continue
                ext = np.column_stack(region.to_pixels(*poly.exterior.xy))
                if len(ext) >= 3:
                    draw.polygon([tuple(p) for p in ext], fill=1)
                for hole in poly.interiors:
                    hp = np.column_stack(region.to_pixels(*hole.xy))
                    if len(hp) >= 3:
                        draw.polygon([tuple(p) for p in hp], fill=0)
        out[np.asarray(canvas, dtype=bool)] = level
    return out


# --------------------------------------------------------------------------------------------------------------
# Stain analysis
# --------------------------------------------------------------------------------------------------------------
def stain_separation(rgb, block_rows=2048):
    """Colour deconvolution (Ruifrok & Johnston) into haematoxylin and DAB optical-density channels.

    Uses scikit-image's standard H-DAB stain vectors and processes the image in row blocks to bound memory.
    Returns (hematoxylin, dab) as float32 arrays; higher = more stain.
    """
    from skimage.color import rgb2hed

    h = np.empty(rgb.shape[:2], dtype=np.float32)
    dab = np.empty(rgb.shape[:2], dtype=np.float32)
    for r in range(0, rgb.shape[0], block_rows):
        hed = rgb2hed(rgb[r:r + block_rows])
        h[r:r + block_rows] = hed[..., 0]
        dab[r:r + block_rows] = hed[..., 2]
    return h, dab


def tile_reduce(arr, tile, func="mean"):
    """Reduce a 2-D array over non-overlapping tile x tile blocks (edges cropped). func: mean, max, sum."""
    h, w = arr.shape[:2]
    th, tw = h // tile, w // tile
    blocks = arr[:th * tile, :tw * tile].reshape(th, tile, tw, tile)
    return getattr(blocks, func)(axis=(1, 3))


def tile_labels(annotations, grid, preview_region, min_fraction=0.5):
    """Assign every tile of a `tile_features` grid an annotation grade.

    The annotations are rasterised at the preview resolution and summarised per tile:
        annotated_fraction  fraction of the tile covered by any annotation polygon
        label               dominant grade among annotated pixels if annotated_fraction >= min_fraction, else 0
    Returns (label, annotated_fraction), both shaped like the grid.
    """
    factor = int(round(grid.downsample / preview_region.downsample))
    raster = rasterize_annotations(annotations, preview_region)
    n_rows, n_cols = grid.shape
    n_grades = int(raster.max())
    per_grade = np.stack([tile_reduce((raster == k).astype(np.float32), factor)[:n_rows, :n_cols]
                          for k in range(1, n_grades + 1)]) if n_grades else np.zeros((1, n_rows, n_cols))
    annotated_fraction = per_grade.sum(axis=0)
    label = np.where(annotated_fraction >= min_fraction, per_grade.argmax(axis=0) + 1, 0)
    return label.astype(np.int8), annotated_fraction.astype(np.float32)


# --------------------------------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------------------------------
# Fixed colour per protein (categorical order validated for colour-vision deficiency); never re-assigned by rank.
PROTEIN_COLORS = {
    "GFAP": "#2a78d6", "CD68": "#eb6834", "CD34": "#1baf7a", "HIF1a": "#eda100",
    "FIB": "#e87ba4", "APP": "#008300", "GAP43": "#4a3aa7", "HE": "#e34948",
}
# Ordinal single-hue ramp for annotation grades (0 = not annotated).
GRADE_COLORS = {0: "#e6e5e0", 1: "#86b6ef", 2: "#2a78d6", 3: "#104281"}


def set_plot_style():
    """Matplotlib defaults shared by the C-Stroke notebooks: light surface, recessive axes, readable type."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 150, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#b5b4ae", "axes.labelcolor": "#2b2b29", "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 10, "image.interpolation": "nearest",
    })


def draw_annotation_outlines(ax, annotations, region, colors=None, linewidth=0.8, alpha=1.0):
    """Draw annotation outlines on an axes showing an image fetched for `region` (pixel coordinates).

    colors  dict grade -> colour (defaults to a light-to-dark ramp).
    """
    colors = colors or GRADE_COLORS
    for rec in annotations:
        for poly in _polygons(rec["geometry"]):
            if poly.geom_type != "Polygon":
                continue
            c, r = region.to_pixels(*poly.exterior.xy)
            ax.plot(c, r, color=colors.get(rec["grade"], GRADE_COLORS[3]), lw=linewidth, alpha=alpha)


def scale_bar(ax, region, length_mm=5.0, loc="lower right", color="black"):
    """Add a scale bar of `length_mm` to an axes showing an image fetched for `region`."""
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar as _Bar

    px = length_mm * 1000.0 / region.mpp
    label = f"{length_mm:g} mm" if length_mm >= 1 else f"{length_mm * 1000:g} µm"
    ax.add_artist(_Bar(ax.transData, px, label, loc, pad=0.4, color=color, frameon=False, size_vertical=px / 40))

