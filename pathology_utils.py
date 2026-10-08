"""pathology_utils - C-Stroke pathology annotations and stain measurements.

The C-Stroke annotations are region polygons drawn on whole-slide images:

    * IHC sections   graded regions of the stained marker, e.g. "sparse / moderately dense / dense astrocytes"
                     on GFAP slides, or single-class regions such as "blood brain barrier disruption" (fibrinogen)
    * H&E sections   the infarcted region

Classes
-------
    SectionAnnotation   the annotation of one slide: polygons with class and grade, measurements, masks,
                        distance maps, filtering, GeoJSON export
    AnnotationSeries    every annotated section of one marker in one block: areas along the block, volume,
                        a voxel stack for 3D reconstruction

The grade of a polygon is the position of its class in slide_utils.MARKERS[marker]["classes"] (1 = mildest).
Geometry is in level-0 canvas pixels (see slide_utils), with y pointing down as in the images.

Functions
---------
    stain_separation    colour deconvolution of an RGB image into haematoxylin and DAB optical densities
    signed_distance_mm  distance of every pixel to the boundary of a mask, negative inside
"""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from io_utils import get_bytes, safe_name
from slide_utils import MARKERS, MPP_LEVEL0, SlideCatalog, canonical_marker

__all__ = ["SectionAnnotation", "AnnotationSeries", "stain_separation", "signed_distance_mm",
           "SECTION_THICKNESS_MM"]

SECTION_THICKNESS_MM = 0.02      # one section number corresponds to 20 um along the block
_UM2_PER_PX = MPP_LEVEL0 ** 2


def _polygons(geometry):
    return [g for g in (geometry.geoms if hasattr(geometry, "geoms") else [geometry]) if g.geom_type == "Polygon"]


def _grade(marker, label):
    classes = MARKERS.get(marker, {}).get("classes", [])
    return classes.index(label) + 1 if label in classes else 1


class SectionAnnotation:
    """Annotation polygons of one slide.

    Each record is a dict with `label` (class name), `grade` (1 = mildest class) and `geometry` (shapely,
    level-0 pixels). Create instances with SectionAnnotation.from_url or SlideHelper.get_annotation.
    """

    def __init__(self, records, marker, block=None, section=None, url=None):
        self.records = list(records)
        self.marker = marker
        self.block = block
        self.section = section
        self.url = url

    @classmethod
    def from_url(cls, url, block=None, section=None, use_cache=True):
        """Download and parse one annotation file of the C-Stroke annotation API."""
        from shapely.geometry import shape
        from shapely.ops import transform

        marker = canonical_marker(url.split("openAtlasJson/")[1].split("/")[0]) if "openAtlasJson/" in url else None
        name = f"annotations/{safe_name(url.split('openAtlasJson/')[-1])}"
        data = get_bytes(url, name, use_cache, validate=lambda b: b.lstrip()[:1] == b"{")
        records = []
        for feature in json.loads(data).get("features", []):
            if not feature.get("geometry"):
                continue
            # Files store (X, Y) with Y negative; flip to image orientation (y down).
            g = transform(lambda x, y, z=None: (x, -np.asarray(y)), shape(feature["geometry"]))
            g = g if g.is_valid else g.buffer(0)
            if g.is_empty:
                continue
            props = (feature.get("properties") or {}).get("data") or {}
            label = props.get("name") or props.get("text") or "annotation"
            records.append({"label": label, "grade": _grade(marker, label), "geometry": g})
        return cls(records, marker, block, section, url)

    def __len__(self):
        return len(self.records)

    def __repr__(self):
        where = f"block {self.block}, section {self.section}, " if self.block is not None else ""
        return f"SectionAnnotation({self.marker}, {where}{len(self)} polygons, {self.area_mm2():.1f} mm2)"

    @property
    def classes(self):
        """Class names of this marker in increasing severity."""
        return MARKERS.get(self.marker, {}).get("classes", [])

    # -- selection -----------------------------------------------------------------------------------------------
    def filter(self, labels=None, min_grade=None, min_area_mm2=None):
        """A new SectionAnnotation with the selected polygons only."""
        keep = []
        for r in self.records:
            if labels is not None and r["label"] not in np.atleast_1d(labels):
                continue
            if min_grade is not None and r["grade"] < min_grade:
                continue
            if min_area_mm2 is not None and r["geometry"].area * _UM2_PER_PX / 1e6 < min_area_mm2:
                continue
            keep.append(r)
        return SectionAnnotation(keep, self.marker, self.block, self.section, self.url)

    def union(self, min_grade=1):
        """Union of all polygons with grade >= min_grade (shapely geometry)."""
        from shapely.ops import unary_union
        return unary_union([r["geometry"] for r in self.records if r["grade"] >= min_grade])

    def by_class(self):
        """{class name: union of its polygons}, in increasing severity."""
        from shapely.ops import unary_union
        labels = [c for c in self.classes if any(r["label"] == c for r in self.records)]
        labels += sorted({r["label"] for r in self.records} - set(labels))
        return {c: unary_union([r["geometry"] for r in self.records if r["label"] == c]) for c in labels}

    # -- measurements --------------------------------------------------------------------------------------------
    def area_mm2(self, min_grade=1):
        """Area covered by polygons with grade >= min_grade (overlaps counted once)."""
        return self.union(min_grade).area * _UM2_PER_PX / 1e6 if self.records else 0.0

    def table(self):
        """One row per polygon: class, grade, area, perimeter, centroid, compactness and number of holes.

        compactness = 4 * pi * area / perimeter**2 (1 for a disc, smaller for elongated or ragged shapes).
        """
        import pandas as pd

        rows = []
        for i, r in enumerate(self.records):
            g = r["geometry"]
            area, perim = g.area * _UM2_PER_PX / 1e6, g.length * MPP_LEVEL0 / 1e3
            c = g.representative_point()
            rows.append({"polygon": i, "class": r["label"], "grade": r["grade"], "area_mm2": area,
                         "perimeter_mm": perim, "x": c.x, "y": c.y,
                         "compactness": 4 * np.pi * area / perim ** 2 if perim else np.nan,
                         "holes": sum(len(p.interiors) for p in _polygons(g))})
        return pd.DataFrame(rows)

    def summary(self):
        """Per class: number of polygons, total and median area (mm2)."""
        t = self.table()
        if t.empty:
            return t
        return (t.groupby(["grade", "class"]).area_mm2
                .agg(polygons="count", total_mm2="sum", median_mm2="median").reset_index())

    # -- rasters -------------------------------------------------------------------------------------------------
    def to_mask(self, region, value="grade"):
        """Rasterise onto the pixel grid of `region` (a slide_utils.Region).

        value  "grade": each pixel holds the highest grade covering it; "binary": 1 inside any polygon
        """
        from PIL import Image, ImageDraw

        h, w = region.shape[:2]
        out = np.zeros((h, w), dtype=np.uint8)
        levels = sorted({r["grade"] for r in self.records}) if value == "grade" else [1]
        for level in levels:
            canvas = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(canvas)
            for r in self.records:
                if value == "grade" and r["grade"] != level:
                    continue
                for poly in _polygons(r["geometry"]):
                    ext = np.column_stack(region.to_pixels(*poly.exterior.xy))
                    draw.polygon([tuple(p) for p in ext], fill=1)
                    for hole in poly.interiors:
                        draw.polygon([tuple(p) for p in np.column_stack(region.to_pixels(*hole.xy))], fill=0)
            out[np.asarray(canvas, dtype=bool)] = level
        return out

    def distance_map(self, region, min_grade=1):
        """Signed distance (mm) of every pixel of `region` to the annotation boundary; negative inside."""
        inside = self.filter(min_grade=min_grade).to_mask(region, "binary").astype(bool)
        return signed_distance_mm(inside, region.mpp)

    # -- export --------------------------------------------------------------------------------------------------
    def to_geojson(self, path=None):
        """GeoJSON FeatureCollection in level-0 image pixels (y down); written to `path` when given."""
        from shapely.geometry import mapping

        coll = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": mapping(r["geometry"]),
             "properties": {"marker": self.marker, "class": r["label"], "grade": r["grade"],
                            "block": self.block, "section": self.section}} for r in self.records]}
        if path is not None:
            with open(path, "w") as f:
                json.dump(coll, f)
        return coll


class AnnotationSeries:
    """All annotated sections of one marker in one block.

    block    tissue block id
    marker   IHC marker ("CD68", "GFAP", ...) or "HE" for the infarct outlined on H&E sections
    """

    def __init__(self, block, marker, catalog=None, use_cache=True):
        self.block = int(block)
        self.marker = canonical_marker(marker)
        self.catalog = catalog if catalog is not None else SlideCatalog(use_cache=use_cache)
        stain = "HE" if self.marker == "HE" else "IHC"
        self._slides = self.catalog.find(self.block, stain, None if stain == "HE" else self.marker, annotated=True)
        self.use_cache = use_cache
        self.annotations = {}

    def __repr__(self):
        state = f"{len(self.annotations)} loaded" if self.annotations else "not loaded"
        return f"AnnotationSeries(block {self.block}, {self.marker}, {len(self._slides)} sections, {state})"

    @property
    def sections(self):
        return sorted(self._slides.section.tolist())

    def load(self, workers=8):
        """Download every annotation of the series in parallel. Unavailable files are skipped and reported."""
        def one(row):
            try:
                return row.section, SectionAnnotation.from_url(row.annotation_url, self.block, row.section,
                                                               self.use_cache)
            except (ValueError, FileNotFoundError, RuntimeError):
                return row.section, None
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(one, [r for _, r in self._slides.iterrows()]))
        self.annotations = {s: a for s, a in sorted(results) if a is not None}
        self.missing = [s for s, a in results if a is None]
        return self

    def __getitem__(self, section):
        return self.annotations[section]

    def table(self):
        """One row per section: position along the block, polygons, and annotated area per class and in total."""
        import pandas as pd

        if not self.annotations:
            self.load()
        rows = []
        for section, ann in self.annotations.items():
            row = {"section": section, "position_mm": section * SECTION_THICKNESS_MM, "polygons": len(ann),
                   "area_mm2": ann.area_mm2()}
            for label, geometry in ann.by_class().items():
                row[label] = geometry.area * _UM2_PER_PX / 1e6
            rows.append(row)
        return pd.DataFrame(rows).fillna(0.0)

    def volume_mm3(self, min_grade=1):
        """Volume estimate by integrating the annotated area over section position (trapezoid rule).

        Accurate when the annotated sections sample the structure densely relative to its changes in shape;
        gaps between annotated sections are interpolated linearly.
        """
        if not self.annotations:
            self.load()
        secs = sorted(self.annotations)
        if len(secs) < 2:
            return 0.0
        z = np.array(secs) * SECTION_THICKNESS_MM
        a = np.array([self.annotations[s].area_mm2(min_grade) for s in secs])
        from scipy.integrate import trapezoid
        return float(trapezoid(a, z))

    def to_stack(self, region, min_grade=1):
        """Binary masks of every loaded section on a common in-plane grid `region`.

        Returns (stack, positions_mm) with stack shaped (n_sections, rows, cols).
        """
        if not self.annotations:
            self.load()
        secs = sorted(self.annotations)
        stack = np.stack([self.annotations[s].filter(min_grade=min_grade).to_mask(region, "binary") for s in secs])
        return stack.astype(bool), np.array(secs) * SECTION_THICKNESS_MM

    def interpolate(self, region, z_step_mm=0.25, min_grade=1):
        """Shape-based interpolation between the annotated sections.

        The signed distance map of each section's annotation is interpolated linearly along the block, and the
        zero level is taken as the boundary. This gives smooth shapes between sections that are far apart.
        Returns (volume, positions_mm): a boolean array (n_positions, rows, cols) on `region`'s grid.
        """
        stack, z = self.to_stack(region, min_grade)
        distance = np.stack([signed_distance_mm(m, region.mpp) for m in stack])
        positions = np.arange(z.min(), z.max() + 1e-9, z_step_mm)
        volume = np.empty((len(positions),) + stack.shape[1:], dtype=bool)
        for n, zi in enumerate(positions):
            j = int(np.clip(np.searchsorted(z, zi), 1, len(z) - 1))
            w = (zi - z[j - 1]) / (z[j] - z[j - 1])
            volume[n] = (1 - w) * distance[j - 1] + w * distance[j] < 0
        return volume, positions

    def to_mesh(self, region, z_step_mm=0.25, min_grade=1, step_size=1):
        """Triangle surface of the interpolated annotation volume (marching cubes).

        Returns (vertices, faces); vertex coordinates are in mm as (position along the block, row, column).
        step_size > 1 gives a coarser, lighter mesh.
        """
        from skimage import measure

        volume, positions = self.interpolate(region, z_step_mm, min_grade)
        padded = np.pad(volume, 1).astype(np.float32)
        verts, faces, _, _ = measure.marching_cubes(padded, 0.5, spacing=(z_step_mm, region.mpp / 1000,
                                                                           region.mpp / 1000),
                                                    step_size=step_size)
        verts -= np.array([z_step_mm, region.mpp / 1000, region.mpp / 1000])       # undo the padding
        verts[:, 0] += positions[0]
        return verts, faces


# ----------------------------------------------------------------------------------------------------------------
# Image measurements
# ----------------------------------------------------------------------------------------------------------------
def stain_separation(rgb, block_rows=2048):
    """Colour deconvolution (Ruifrok and Johnston, 2001) into haematoxylin and DAB optical densities.

    Uses scikit-image's reference haematoxylin-DAB vectors and processes the image in row blocks to bound memory.
    The C-Stroke IHC slides carry no haematoxylin counterstain, so nearly all optical density appears in DAB.
    Returns (haematoxylin, dab) as float32 arrays; higher means more stain.
    """
    from skimage.color import rgb2hed

    hema = np.empty(rgb.shape[:2], dtype=np.float32)
    dab = np.empty(rgb.shape[:2], dtype=np.float32)
    for r in range(0, rgb.shape[0], block_rows):
        hed = rgb2hed(rgb[r:r + block_rows])
        hema[r:r + block_rows], dab[r:r + block_rows] = hed[..., 0], hed[..., 2]
    return hema, dab


def signed_distance_mm(mask, um_per_px):
    """Euclidean distance (mm) of every pixel to the boundary of `mask`: negative inside, positive outside."""
    from scipy import ndimage as ndi

    mask = np.asarray(mask, dtype=bool)
    inside = ndi.distance_transform_edt(mask)
    outside = ndi.distance_transform_edt(~mask)
    return np.where(mask, -inside, outside) * um_per_px / 1000.0
