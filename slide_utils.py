"""slide_utils - the C-Stroke slide catalogue and whole-slide image access.

Two classes cover the histology side of the C-Stroke dataset (case HB02CV):

    SlideCatalog   what exists: blocks, stains, sections, IHC markers, annotation files, and the "sets" that group
                   each immunohistochemistry (IHC) section with its nearest Nissl, H&E and myelin sections.
    SlideHelper    image access for one block and stain: section lists, pyramid levels, whole sections at any
                   resolution, regions and fields of view, tiles for analysis, block-face photographs.

Typical use
-----------
    from slide_utils import SlideCatalog, SlideHelper

    catalog = SlideCatalog()
    catalog.summary()                                    # blocks, stains and annotation coverage
    helper = SlideHelper(block=313, stain="N", downsample=6, catalog=catalog)
    sections = helper.get_section_numbers()
    image, region = helper.get_sectionimage(sections[100])

Coordinates and resolution
--------------------------
Every slide is stored on a common 192,000 x 192,000 pixel canvas that is registered within its block, so a
coordinate on one stain points at the same tissue on the neighbouring sections of the other stains. Positions are
given in full-resolution ("level-0") canvas pixels: x to the right, y down, origin at the top-left corner.

`downsample` is a power-of-two exponent: an image fetched with downsample=d has 2**d level-0 pixels per output
pixel. The pyramid has 8 levels, from the full canvas (d=0) to a 1,500 px overview (d=7).

MPP_LEVEL0 = 1.0 um per level-0 pixel. The value is not stored with the images; it was derived from the C-Stroke
reference space: the tissue area of sections measured on the slides matches the area of the same planes in the
ex-vivo MRI, the reconstructed histology volume and the atlas labels at 0.97-1.02 um per pixel, and the infarct
areas listed in the platform's manifest correspond to 1.04 um per pixel. Treat physical units as accurate to a few
per cent.
"""

import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import quote

import numpy as np

from io_utils import BASE_URL, BLOCKFACE_URL, IIP_URL, SETS_URL, get, get_bytes, safe_name

__all__ = ["SLIDE_SIZE", "MPP_LEVEL0", "N_LEVELS", "STAINS", "BLOCKS", "MARKERS", "Region",
           "SlideCatalog", "SlideHelper", "canonical_stain", "canonical_marker", "tissue_mask"]

SLIDE_SIZE = 192000          # level-0 width and height of every slide canvas (px)
MPP_LEVEL0 = 1.0             # microns per level-0 pixel (derived from the reference space; see module docstring)
N_LEVELS = 8                 # pyramid levels: 1500 * 2**l px, l = 0..7
_MAX_REQUEST_PX = 4096       # larger outputs are fetched in parallel chunks

STAINS = {"IHC": "Immunohistochemistry", "N": "Nissl", "HE": "H&E", "M": "Myelin"}

# Tissue blocks of case HB02CV, with the position given in the C-Stroke reference-space manifest.
BLOCKS = {217: "right medial", 218: "right lateral", 313: "left medial", 314: "left lateral", 315: "left mid"}

# IHC markers stained in the dataset: what each one shows and its annotation classes in increasing severity.
# "HE" denotes the infarct outlined on H&E sections.
MARKERS = {
    "CD68":  {"target": "Microglia and macrophages",
              "classes": ["moderately dense CD68+ immune cells", "dense CD68+ immune cells"]},
    "GFAP":  {"target": "Reactive astrocytes",
              "classes": ["sparse astrocytes", "moderately dense astrocytes", "dense astrocytes"]},
    "CD34":  {"target": "Microvessels",
              "classes": ["sparse microvessels", "moderately dense microvessels", "dense microvessels"]},
    "HIF1a": {"target": "Hypoxic cells", "classes": ["moderately dense HIF1α+ cells", "dense HIF1α+ cells"]},
    "FIB":   {"target": "Fibrinogen leakage (blood-brain barrier disruption)",
              "classes": ["blood brain barrier disruption"]},
    "APP":   {"target": "Axonal injury", "classes": ["axonal injury"]},
    "GAP43": {"target": "Axonal regrowth", "classes": ["axonal regrowth"]},
    "AQP4":  {"target": "Aquaporin-4 (astrocytic end-feet)", "classes": []},
    "HE":    {"target": "Infarct (H&E)", "classes": ["Infarcted region"]},
}

_STAIN_ALIASES = {"IHC": "IHC", "IHCS": "IHC", "IHC3": "IHC", "N": "N", "NISL": "N", "NISSL": "N",
                  "HE": "HE", "H&E": "HE", "HEOS": "HE", "M": "M", "MYEL": "M", "MYELIN": "M"}
_MARKER_ALIASES = {"CD68": "CD68", "GFAP": "GFAP", "CD34": "CD34", "AQP4": "AQP4", "GAP43": "GAP43",
                   "HIF1A": "HIF1a", "H1F1A": "HIF1a", "HIFIA": "HIF1a", "FIB": "FIB", "FIBRINOGEN": "FIB",
                   "FIBRINOGER": "FIB", "APP": "APP", "APP+": "APP", "HEOS": "HE", "HE": "HE"}


def canonical_stain(stain):
    """'Nissl', 'NISL', 'n' -> 'N'; 'H&E', 'HEOS' -> 'HE'; and so on."""
    key = str(stain).strip().upper()
    if key not in _STAIN_ALIASES:
        raise ValueError(f"Unknown stain {stain!r}; use one of {list(STAINS)}.")
    return _STAIN_ALIASES[key]


def canonical_marker(marker):
    """Spellings used in file names and annotation folders ('HIF1A', 'Fibrinogen', ...) -> canonical key."""
    if marker is None:
        return None
    return _MARKER_ALIASES.get(str(marker).strip().upper(), str(marker).strip())


# ----------------------------------------------------------------------------------------------------------------
# Regions
# ----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Region:
    """Where a fetched image lies on the slide canvas: level-0 origin, level-0 pixels per image pixel, shape."""
    x0: float
    y0: float
    scale: float
    shape: tuple

    @property
    def mpp(self):
        """Microns per pixel of the image (based on MPP_LEVEL0)."""
        return MPP_LEVEL0 * self.scale

    @property
    def bounds(self):
        """(x0, y0, x1, y1) in level-0 pixels."""
        h, w = self.shape[:2]
        return self.x0, self.y0, self.x0 + w * self.scale, self.y0 + h * self.scale

    def to_pixels(self, x, y):
        """Level-0 coordinates -> (col, row) of this image."""
        return (np.asarray(x) - self.x0) / self.scale, (np.asarray(y) - self.y0) / self.scale

    def to_level0(self, col, row):
        """(col, row) of this image -> level-0 coordinates."""
        return self.x0 + np.asarray(col) * self.scale, self.y0 + np.asarray(row) * self.scale


def _decode_jpeg(data):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def _is_image(data):
    return data[:2] == b"\xff\xd8"


def tissue_mask(rgb, od_threshold=0.03, sigma=1.0, min_object_frac=1e-4):
    """Boolean tissue mask from optical density. The slide background is pure white, so any stain raises OD."""
    from scipy import ndimage as ndi

    od = -np.log10(np.clip(rgb.astype(np.float32) + 1.0, 1.0, 256.0) / 256.0).sum(axis=-1)
    if sigma:
        od = ndi.gaussian_filter(od, sigma)
    mask = ndi.binary_fill_holes(od > od_threshold)
    labels, n = ndi.label(mask)
    if n:
        sizes = ndi.sum(mask, labels, index=np.arange(1, n + 1))
        mask = np.isin(labels, np.flatnonzero(sizes >= min_object_frac * mask.size) + 1)
    return mask


# ----------------------------------------------------------------------------------------------------------------
# Catalogue
# ----------------------------------------------------------------------------------------------------------------
class SlideCatalog:
    """The C-Stroke slide catalogue.

    Built from the platform's `sets_data.json`. Each of its 505 sets anchors one IHC section and lists the nearest
    Nissl, H&E and myelin sections of the same block. A routine-stain section is often the nearest neighbour of
    several IHC sections and then belongs to each of those sets.

    Attributes
        entries   one row per (set, slide): set membership and the slide's offset from the set's IHC section
        slides    one row per distinct slide (block, stain, section)
    """

    def __init__(self, use_cache=True):
        import pandas as pd

        raw = json.loads(get_bytes(SETS_URL, "sets_data.json", use_cache))
        rows = []
        for entry in raw["sections"]:
            for slot in range(1, 5):
                rec = entry.get(f"section_{slot}")
                if not rec or not rec.get("path"):
                    continue
                stain = canonical_stain(rec["stain"])
                marker = None
                if stain == "IHC":
                    m = re.search(r"\]\[([^\]]+)\]", rec["path"])
                    marker = canonical_marker(m.group(1)) if m else None
                ann = rec.get("annotation") or None
                # The catalogue also attaches each set's H&E infarct file to its Nissl entry (for overlay in the
                # viewer). An annotation is kept here only for the slide it was drawn on.
                m = re.search(r"_SE-(\d+)_registered", ann) if ann else None
                if ann and (m is None or int(m.group(1)) != int(rec["secID"])):
                    ann = None
                rows.append({
                    "set_id": entry["index"], "slot": slot,
                    "block": int(rec["bid"]), "section": int(rec["secID"]), "stain": stain, "marker": marker,
                    "path": rec["path"],
                    "annotation_url": f"{BASE_URL}/{ann.lstrip('/')}" if ann else None,
                    "annotation_type": canonical_marker(ann.split("/")[2]) if ann else None,
                    "z_mm": rec.get("z_mm"), "offset_mm": rec.get("dist_from_anchor"),
                    "global_id": rec.get("globalID"),
                })
        self.entries = pd.DataFrame(rows)
        self.slides = (self.entries.drop(columns=["set_id", "slot", "offset_mm"])
                       .drop_duplicates(["block", "stain", "section"])
                       .sort_values(["block", "stain", "section"]).reset_index(drop=True))

    def __repr__(self):
        return (f"SlideCatalog({self.entries.set_id.nunique()} sets, {len(self.slides)} slides, "
                f"blocks {self.blocks})")

    def __len__(self):
        return len(self.slides)

    @property
    def blocks(self):
        return sorted(self.slides.block.unique().tolist())

    def summary(self):
        """One row per block: description, section range and number of slides per stain and annotation type."""
        import pandas as pd

        s = self.slides
        table = s.pivot_table(index="block", columns="stain", values="section", aggfunc="count", fill_value=0)
        table = table.reindex(columns=list(STAINS), fill_value=0).rename(columns=STAINS)
        table.insert(0, "position", [BLOCKS.get(b, "") for b in table.index])
        table.insert(1, "sections", [f"{s[s.block == b].section.min()}-{s[s.block == b].section.max()}"
                                     for b in table.index])
        table["annotated IHC"] = s[(s.stain == "IHC") & s.annotation_url.notna()].groupby("block").size()
        table["infarct-annotated H&E"] = s[(s.stain == "HE") & s.annotation_url.notna()].groupby("block").size()
        return table.fillna(0).astype({c: int for c in table.columns if c not in ("position", "sections")})

    def markers(self, annotated=None):
        """IHC slides per marker and block (only slides with an annotation file when annotated=True)."""
        s = self.slides[self.slides.stain == "IHC"]
        if annotated is not None:
            s = s[s.annotation_url.notna() == bool(annotated)]
        return s.pivot_table(index="marker", columns="block", values="section", aggfunc="count",
                             fill_value=0).reindex([m for m in MARKERS if m in set(s.marker)])

    def find(self, block=None, stain=None, marker=None, annotated=None):
        """Filter the distinct slides by block(s), stain, IHC marker and presence of an annotation file."""
        s = self.slides
        if block is not None:
            s = s[s.block.isin(np.atleast_1d(block))]
        if stain is not None:
            s = s[s.stain == canonical_stain(stain)]
        if marker is not None:
            s = s[s.marker == canonical_marker(marker)]
        if annotated is not None:
            s = s[s.annotation_url.notna() == bool(annotated)]
        return s.reset_index(drop=True)

    def get_section_numbers(self, block, stain, marker=None, annotated=None):
        """Sorted section numbers of one block and stain (optionally one IHC marker, or annotated slides only)."""
        return sorted(self.find(block, stain, marker, annotated).section.tolist())

    def get_slide(self, block, stain, section):
        """Catalogue row of one slide; raises KeyError when it does not exist."""
        s = self.find(block, stain)
        row = s[s.section == int(section)]
        if row.empty:
            raise KeyError(f"No {STAINS[canonical_stain(stain)]} slide for block {block}, section {section}.")
        return row.iloc[0]

    def get_set(self, set_id):
        """All slides of one set, IHC anchor first."""
        return self.entries[self.entries.set_id == set_id].sort_values("slot").reset_index(drop=True)

    def nearest(self, block, section, stain, marker=None, annotated=None):
        """The slide of `stain` closest to `section` in the same block. Adds `gap_sections` (20 um each)."""
        s = self.find(block, stain, marker, annotated)
        if s.empty:
            raise KeyError(f"No matching {stain} slides in block {block}.")
        row = s.iloc[int(np.argmin(np.abs(s.section.to_numpy() - int(section))))].copy()
        row["gap_sections"] = int(row.section) - int(section)
        return row

    def neighbours(self, block, section, max_gap=None):
        """The nearest slide of every stain (and every IHC marker) around a section.

        max_gap  optional limit on the distance in sections (one section = 20 um)
        """
        import pandas as pd

        rows = []
        for stain in STAINS:
            groups = [None] if stain != "IHC" else sorted(self.find(block, "IHC").marker.dropna().unique())
            for marker in groups:
                try:
                    r = self.nearest(block, section, stain, marker)
                except KeyError:
                    continue
                if max_gap is None or abs(r.gap_sections) <= max_gap:
                    rows.append(r)
        table = pd.DataFrame(rows)
        table["distance_um"] = table.gap_sections.abs() * 20
        return table.sort_values(["distance_um", "stain"]).reset_index(drop=True)

    def to_listing(self, block=None):
        """Download table: one row per slide with its image, annotation and block-face URLs."""
        s = self.find(block).copy()
        s["thumbnail_url"] = [f"{IIP_URL}?FIF={quote(p)}&WID=2048&CVT=jpeg" for p in s.path]
        s["zoomify_url"] = [f"{IIP_URL}?Zoomify={quote(p)}/ImageProperties.xml" for p in s.path]
        s["blockface_url"] = [f"{BLOCKFACE_URL}/{b}/B_{b}-ST_BFIW-SE_{sec}.jpg"
                              for b, sec in zip(s.block, s.section)]
        cols = ["block", "stain", "section", "marker", "z_mm", "global_id", "path", "thumbnail_url",
                "zoomify_url", "annotation_url", "annotation_type", "blockface_url"]
        return s[cols]


# ----------------------------------------------------------------------------------------------------------------
# Image access
# ----------------------------------------------------------------------------------------------------------------
class SlideHelper:
    """Whole-slide image access for one block and one stain.

    block        tissue block id (217, 218, 313, 314, 315)
    stain        "IHC", "N" (Nissl), "HE" (H&E) or "M" (myelin); common spellings are accepted
    downsample   default resolution exponent: images have 2**downsample level-0 pixels per pixel
                 (0 = full resolution, about 1 um/px; 6 = about 64 um/px)
    catalog      an existing SlideCatalog (otherwise one is created)
    """

    def __init__(self, block, stain="N", downsample=6, catalog=None, use_cache=True):
        self.block = int(block)
        self.stain = canonical_stain(stain)
        if not 0 <= downsample < N_LEVELS:
            raise ValueError(f"downsample must be between 0 and {N_LEVELS - 1}.")
        self.downsample = int(downsample)
        self.catalog = catalog if catalog is not None else SlideCatalog(use_cache=use_cache)
        self.use_cache = use_cache
        self._slides = self.catalog.find(self.block, self.stain).set_index("section")
        if self._slides.empty:
            raise ValueError(f"No {STAINS[self.stain]} slides in block {self.block}.")
        self._bbox_cache = {}

    def __repr__(self):
        return (f"SlideHelper(block={self.block} [{BLOCKS.get(self.block, '')}], stain={self.stain} "
                f"[{STAINS[self.stain]}], {len(self._slides)} sections, downsample={self.downsample} "
                f"(~{self.mpp:g} um/px))")

    @property
    def mpp(self):
        """Approximate microns per pixel at the default downsample."""
        return MPP_LEVEL0 * 2 ** self.downsample

    # -- sections ------------------------------------------------------------------------------------------------
    def get_section_numbers(self, marker=None, annotated=None):
        """Sorted section numbers (IHC: optionally one marker; annotated=True keeps annotated slides only)."""
        s = self._slides
        if marker is not None:
            s = s[s.marker == canonical_marker(marker)]
        if annotated is not None:
            s = s[s.annotation_url.notna() == bool(annotated)]
        return sorted(s.index.tolist())

    def get_slide(self, section):
        """Catalogue row of a section."""
        if int(section) not in self._slides.index:
            raise KeyError(f"Section {section} has no {STAINS[self.stain]} slide in block {self.block}.")
        return self._slides.loc[int(section)]

    def get_path(self, section):
        """Path of the section's image on the C-Stroke image server."""
        return self.get_slide(section).path

    # -- image metadata ------------------------------------------------------------------------------------------
    def get_levels(self, section):
        """Pyramid levels reported by the image server, with the approximate resolution of each."""
        import pandas as pd

        text = get(f"{IIP_URL}?FIF={quote(self.get_path(section))}&obj=Resolutions").text
        sizes = [tuple(int(v) for v in s.split()) for s in text.split(":", 1)[1].strip().split(",")]
        rows = [{"downsample": N_LEVELS - 1 - i, "width_px": w, "height_px": h,
                 "um_per_px": MPP_LEVEL0 * SLIDE_SIZE / w} for i, (w, h) in enumerate(sizes)]
        return pd.DataFrame(rows).sort_values("downsample").reset_index(drop=True)

    def get_tissue_bbox(self, section, margin_px=2000):
        """Level-0 bounding box (x0, y0, width, height) of the tissue, padded by `margin_px`."""
        if section in self._bbox_cache:
            return self._bbox_cache[section]
        thumb, reg = self.get_region(section, 0, 0, SLIDE_SIZE, SLIDE_SIZE, downsample=N_LEVELS - 1)
        mask = tissue_mask(thumb)
        if not mask.any():
            raise ValueError(f"No tissue detected on block {self.block}, section {section} ({self.stain}).")
        rows, cols = np.nonzero(mask)
        x0, y0 = reg.to_level0(cols.min(), rows.min())
        x1, y1 = reg.to_level0(cols.max() + 1, rows.max() + 1)
        x0, y0 = max(0.0, x0 - margin_px), max(0.0, y0 - margin_px)
        x1, y1 = min(float(SLIDE_SIZE), x1 + margin_px), min(float(SLIDE_SIZE), y1 + margin_px)
        self._bbox_cache[section] = (x0, y0, x1 - x0, y1 - y0)
        return self._bbox_cache[section]

    def get_tissue_area_mm2(self, section):
        """Tissue area of the section in mm2, measured on the 128 um/px overview."""
        thumb, reg = self.get_region(section, 0, 0, SLIDE_SIZE, SLIDE_SIZE, downsample=N_LEVELS - 1)
        return float(tissue_mask(thumb).sum() * reg.mpp ** 2 / 1e6)

    def get_image_url(self, section, width=2048):
        """URL of a JPEG overview of the whole slide canvas, `width` pixels wide."""
        return f"{IIP_URL}?FIF={quote(self.get_path(section))}&WID={int(width)}&CVT=jpeg"

    # -- image data ----------------------------------------------------------------------------------------------
    def _fetch(self, path, x0, y0, width, height, out_w):
        rgn = ",".join(f"{v / SLIDE_SIZE:.8f}" for v in (x0, y0, width, height))
        url = f"{IIP_URL}?FIF={quote(path)}&RGN={rgn}&WID={out_w}&CVT=jpeg"
        name = f"tiles/{safe_name(path.rsplit('/', 1)[-1])}/{int(x0)}_{int(y0)}_{int(width)}_{int(height)}_{out_w}.jpg"
        try:
            return _decode_jpeg(get_bytes(url, name, self.use_cache, validate=_is_image))
        except ValueError:
            raise RuntimeError(f"Image region not available: {path}") from None

    def get_region(self, section, x0, y0, width, height, downsample=None, workers=8):
        """Image of a level-0 rectangle at 2**downsample level-0 pixels per pixel. Returns (rgb, Region).

        Large outputs are assembled from parallel requests of at most 4096 x 4096 pixels.
        """
        from PIL import Image

        d = self.downsample if downsample is None else downsample
        scale = float(2 ** d)
        path = self.get_path(section)
        out_w, out_h = max(1, int(round(width / scale))), max(1, int(round(height / scale)))
        out = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
        step = _MAX_REQUEST_PX
        jobs = [(r, c) for r in range(0, out_h, step) for c in range(0, out_w, step)]

        def one(job):
            r, c = job
            h, w = min(step, out_h - r), min(step, out_w - c)
            img = self._fetch(path, x0 + c * scale, y0 + r * scale, w * scale, h * scale, w)
            if img.shape[:2] != (h, w):                     # the server can round by one pixel
                img = np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))
            return r, c, img

        if len(jobs) == 1:
            results = [one(jobs[0])]
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(one, jobs))
        for r, c, img in results:
            out[r:r + img.shape[0], c:c + img.shape[1]] = img
        return out, Region(float(x0), float(y0), scale, out.shape[:2])

    def get_sectionimage(self, section, downsample=None, crop=True):
        """The whole section at the given resolution, cropped to the tissue unless crop=False."""
        bbox = self.get_tissue_bbox(section) if crop else (0, 0, SLIDE_SIZE, SLIDE_SIZE)
        return self.get_region(section, *bbox, downsample=downsample)

    def get_field(self, section, center, size_um, downsample=None):
        """Square field of view of `size_um` microns centred on a level-0 point (x, y)."""
        size = size_um / MPP_LEVEL0
        return self.get_region(section, center[0] - size / 2, center[1] - size / 2, size, size, downsample)

    def iter_tiles(self, section, tile_px=224, downsample=1, min_tissue=0.5, bbox=None, chunk_tiles=8):
        """Yield (tile, x0, y0) for non-overlapping tiles of the tissue, for analysis or model input.

        tile_px     tile edge in output pixels (tile covers tile_px * 2**downsample level-0 pixels)
        min_tissue  minimum tissue fraction for a tile to be returned
        bbox        level-0 (x0, y0, width, height) to tile; defaults to the tissue bounding box
        Tiles are fetched in chunks of chunk_tiles x chunk_tiles to keep the number of requests low.
        """
        scale = 2 ** downsample
        tile_l0 = tile_px * scale
        x0, y0, w, h = bbox if bbox is not None else self.get_tissue_bbox(section)
        n_rows, n_cols = int(h // tile_l0), int(w // tile_l0)
        for r in range(0, n_rows, chunk_tiles):
            for c in range(0, n_cols, chunk_tiles):
                nr, nc = min(chunk_tiles, n_rows - r), min(chunk_tiles, n_cols - c)
                cx, cy = x0 + c * tile_l0, y0 + r * tile_l0
                img, _ = self.get_region(section, cx, cy, nc * tile_l0, nr * tile_l0, downsample)
                for i in range(nr):
                    for j in range(nc):
                        tile = img[i * tile_px:(i + 1) * tile_px, j * tile_px:(j + 1) * tile_px]
                        if tile.shape[:2] == (tile_px, tile_px) and tissue_mask(tile, sigma=0).mean() >= min_tissue:
                            yield tile, cx + j * tile_l0, cy + i * tile_l0

    # -- annotations and block-face photographs ------------------------------------------------------------------
    def get_annotation(self, section):
        """Pathology annotation of the section (see pathology_utils.SectionAnnotation), or None if there is none."""
        from pathology_utils import SectionAnnotation

        url = self.get_slide(section).annotation_url
        return SectionAnnotation.from_url(url, block=self.block, section=section) if url else None

    def get_blockface(self, section, search=30):
        """Block-face photograph taken before the section was cut. Returns (rgb, section_used).

        Photographs exist for most, not all, sections; the nearest available one within `search` sections is used.
        """
        for offset in [0] + [s * o for o in range(1, search + 1) for s in (-1, 1)]:
            sec = int(section) + offset
            url = f"{BLOCKFACE_URL}/{self.block}/B_{self.block}-ST_BFIW-SE_{sec}.jpg"
            try:
                data = get_bytes(url, f"blockface/{self.block}/{sec}.jpg", self.use_cache, validate=_is_image)
            except (FileNotFoundError, ValueError):
                continue
            return _decode_jpeg(data), sec
        raise KeyError(f"No block-face photograph within {search} sections of {section} in block {self.block}.")
