"""refspace_utils - the C-Stroke MRI reference space and its link to the histology sections.

The C-Stroke platform shows MRI and histology of case HB02CV in one ex-vivo reference space: a 211 x 377 x 238
voxel grid (0.488 x 0.488 x 0.7 mm). The platform's MRI viewer reads its data from
https://c-stroke.humanbrain.in/stroke_atlas/viewer/data/, and the ReferenceSpace class reads the same files:

    manifest.json         grid geometry, series, lesion masks, atlas label names and the geometry of each tissue
                          block (which histology section passes through which voxel)
    <series>.bin          intensity volumes stored as 8-bit values with a per-volume range (lo, hi)
    lab_mni.bin           anatomical labels (MNI-based parcellation, 125 regions)
    lab_art.bin           arterial territory labels (32 territories)
    segs.bin              lesion masks, one bit per mask
    stains.bin            the stains available at every global section index, one bit per stain

Grid axes:  i runs inferior -> superior, j anterior -> posterior, k left -> right.
Arrays returned by this module are indexed [i, j, k].

Block geometry: for block b, the section coordinate of voxel (i, j, k) is
    bk = krow[0] * i + krow[1] * j + krow[2] * k + krow[3]          (0 <= bk <= nslices - 1 inside the block)
    position_mm = mm_slope * bk + mm_int
and the section number follows from the position in steps of 20 um. The cutting planes are oblique to the grid,
so one histology section crosses several k slices.
"""

import json

import numpy as np

from io_utils import REFSPACE_URL, get_bytes

__all__ = ["ReferenceSpace", "SECTION_THICKNESS_MM"]

SECTION_THICKNESS_MM = 0.02
_SIDE_SWAP = {"left": "right", "right": "left", "Left": "Right", "Right": "Left", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def _swap_sides(name):
    import re
    return re.sub(r"\b(left|right|Left|Right|LEFT|RIGHT)\b", lambda m: _SIDE_SWAP[m.group(0)], name)


class ReferenceSpace:
    """MRI volumes, labels and lesion masks of the C-Stroke reference space, with histology block geometry.

    Volumes are downloaded on first use and cached (about 2-6 MB each, compressed).
    """

    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self.manifest = json.loads(get_bytes(f"{REFSPACE_URL}/manifest.json", "refspace/manifest.json", use_cache))
        space = self.manifest["space"]
        self.shape = tuple(space["dims"])                    # (NI, NJ, NK)
        self.spacing = tuple(space["spacing"])               # mm along i, j, k
        self.crop_origin = tuple(space["crop_origin"])
        self._raw = {}
        self._block_cache = None

    def __repr__(self):
        return (f"ReferenceSpace(grid {self.shape[0]} x {self.shape[1]} x {self.shape[2]} voxels, "
                f"{self.spacing[0]:.3f} x {self.spacing[1]:.3f} x {self.spacing[2]:.2f} mm)")

    # -- catalogue -----------------------------------------------------------------------------------------------
    def series(self):
        """Intensity volumes: id, description, time point, sequence and the blocks they cover."""
        import pandas as pd

        labels = {s["id"]: s for s in self.manifest.get("series", [])}
        backdrops = {b["id"]: b["label"] for b in self.manifest.get("backdrops", [])}
        rows = []
        for name, info in self.manifest["volumes"]["gray"].items():
            s = labels.get(name, {})
            rows.append({"name": name, "description": s.get("label") or backdrops.get(name, name),
                         "timepoint": s.get("tp", "ex vivo" if name.startswith("ex_") else ""),
                         "sequence": s.get("seq", name.split("_")[-1]),
                         "range": (info["lo"], info["hi"]),
                         "blocks": "all" if info.get("blocks") is None else ", ".join(map(str, info["blocks"]))})
        return pd.DataFrame(rows)

    def lesion_masks(self):
        """Lesion masks shipped with the data: id, description and time point."""
        import pandas as pd
        return pd.DataFrame([{"id": s["id"], "description": s["label"], "timepoint": s.get("tp") or "ex vivo"}
                             for s in self.manifest["segs"]])

    def blocks(self):
        """Histology blocks with their position, number of sections and span along the cutting axis (mm)."""
        import pandas as pd

        rows = []
        for bid, b in self.manifest["blocks"].items():
            rows.append({"block": int(bid), "position": self.manifest["block_desc"].get(bid, ""),
                         "sections": f"{b['sec_min']}-{b['sec_max']}",
                         "mm_from": min(b["mm_at_secmin"], b["mm_at_secmax"]),
                         "mm_to": max(b["mm_at_secmin"], b["mm_at_secmax"]),
                         "grid_slices": b["nslices"]})
        return pd.DataFrame(rows).sort_values("mm_from").reset_index(drop=True)

    def infarct_sections(self):
        """H&E sections with an infarct annotation as listed in the manifest.

        Columns: block, section, position along the cutting axis (mm), block_slice (index of the block's own
        slice grid, see blocks()), qc (PASS/FAIL) and infarct_area_mm2 (the infarct area provided with the data).
        """
        import pandas as pd
        t = pd.DataFrame(self.manifest["he_pairs"]).rename(columns={"b": "block", "sec": "section", "mm": "position_mm",
                                                                    "k": "block_slice", "area": "infarct_area_mm2"})
        return t.sort_values(["block", "section"]).reset_index(drop=True)

    # -- volumes -------------------------------------------------------------------------------------------------
    def _bytes(self, name):
        if name not in self._raw:
            data = get_bytes(f"{REFSPACE_URL}/{name}.bin", f"refspace/{name}.bin", self.use_cache)
            if data[:2] == b"\x1f\x8b":                     # served compressed without transparent decoding
                import gzip
                data = gzip.decompress(data)
            self._raw[name] = np.frombuffer(data, dtype=np.uint8)
        return self._raw[name]

    def _grid(self, name):
        ni, nj, nk = self.shape
        a = self._bytes(name)
        if a.size != ni * nj * nk:
            raise ValueError(f"{name}: {a.size} values, expected {ni * nj * nk}")
        return a.reshape(nk, ni, nj).transpose(1, 2, 0)       # stored k-major; return [i, j, k]

    def volume(self, name, mask_missing=True):
        """Intensity volume in its original units (float32, [i, j, k]).

        mask_missing  voxels outside the blocks a series was acquired in, and empty there, are set to NaN, as the
                      platform viewer treats them as missing data rather than measured intensity
        """
        info = self.manifest["volumes"]["gray"][name]
        u8 = self._grid(name)
        vol = info["lo"] + u8.astype(np.float32) / 255.0 * (info["hi"] - info["lo"])
        if mask_missing and info.get("blocks"):
            owner, _ = self.block_map()
            vol[(u8 == 0) & ~np.isin(owner, info["blocks"])] = np.nan
        return vol

    def labels(self, atlas="mni"):
        """Label volume (uint8, [i, j, k]) of 'mni' (anatomical regions) or 'art' (arterial territories)."""
        return self._grid({"mni": "lab_mni", "art": "lab_art"}[atlas])

    def label_names(self, atlas="mni", swap_arterial_sides=True):
        """Label table: id, name and display colour.

        The arterial atlas as delivered is mirrored in laterality (manifest notes); with swap_arterial_sides=True
        its left/right words are swapped so that names match the subject's sides, as in the platform viewer.
        """
        import pandas as pd

        rows = []
        for key, entry in self.manifest["atlas"][atlas].items():
            name, r, g, b = entry[:4]
            row = {"id": int(key), "name": name, "color": f"#{r:02x}{g:02x}{b:02x}"}
            if atlas == "art":                                  # arterial entries add a group and an acronym
                acronym = entry[5]
                if swap_arterial_sides:
                    name = _swap_sides(name)
                    acronym = acronym[:-1] + {"L": "R", "R": "L"}.get(acronym[-1], acronym[-1])
                row["name"], row["group"], row["acronym"] = name, entry[4], acronym
            rows.append(row)
        return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)

    def lesion_mask(self, mask_id):
        """Boolean lesion mask ([i, j, k]) by id, e.g. 'iv11_dwi_hyper' (see lesion_masks())."""
        bit = self.manifest["volumes"]["segs"]["bits"][mask_id]
        return (self._grid("segs") >> bit & 1).astype(bool)

    def stains_by_global_section(self):
        """Stains available at each global section index (one row per index with at least one stain)."""
        import pandas as pd

        bits = self._bytes(self.manifest.get("stains_file", "stains.bin").rsplit(".", 1)[0])
        names = self.manifest["stains"]
        idx = np.flatnonzero(bits)
        return pd.DataFrame({"global_id": idx, **{n: (bits[idx] >> i & 1).astype(bool) for i, n in enumerate(names)}})

    # -- geometry ------------------------------------------------------------------------------------------------
    def voxel_mm(self, i, j, k):
        """Reference-space coordinates (mm) of voxel indices."""
        return tuple((np.asarray(v) + o) * s for v, o, s in zip((i, j, k), self.crop_origin, self.spacing))

    def _block(self, block):
        return self.manifest["blocks"][str(int(block))]

    def block_map(self):
        """Owner block of every voxel and its section coordinate bk within that block.

        Returns (owner, bk): owner holds the block id, or 0 for voxels outside every block; bk is NaN there.
        As in the platform viewer, a voxel belongs to the block in which it lies deepest.
        """
        if self._block_cache is None:
            i, j, k = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in self.shape], indexing="ij")
            best_margin = np.full(self.shape, -np.inf, dtype=np.float32)
            owner = np.zeros(self.shape, dtype=np.int16)
            bk_best = np.full(self.shape, np.nan, dtype=np.float32)
            for bid, b in self.manifest["blocks"].items():
                r = b["krow"]
                bk = r[0] * i + r[1] * j + r[2] * k + r[3]
                margin = np.minimum(bk, b["nslices"] - 1 - bk)
                better = margin > best_margin
                best_margin[better], owner[better], bk_best[better] = margin[better], int(bid), bk[better]
            inside = best_margin >= 0
            owner[~inside], bk_best[~inside] = 0, np.nan
            self._block_cache = (owner, bk_best)
        return self._block_cache

    def section_position_mm(self, block, section):
        """Position (mm) along the cutting axis of a histology section."""
        b = self._block(block)
        direction = 1 if b["mm_at_secmax"] >= b["mm_at_secmin"] else -1
        return b["mm_at_secmin"] + direction * (int(section) - b["sec_min"]) * SECTION_THICKNESS_MM

    def section_at(self, i, j, k):
        """(block, section number) of the histology section passing through voxel (i, j, k), or None outside."""
        owner, bk = self.block_map()
        bid = int(owner[i, j, k])
        if bid == 0:
            return None
        b = self._block(bid)
        mm = b["mm_slope"] * bk[i, j, k] + b["mm_int"]
        direction = 1 if b["mm_at_secmax"] >= b["mm_at_secmin"] else -1
        sec = b["sec_min"] + direction * round((mm - b["mm_at_secmin"]) / SECTION_THICKNESS_MM)
        return bid, int(min(max(sec, b["sec_min"]), b["sec_max"]))

    def section_plane(self, block, section):
        """k index of the section plane for every (i, j) column: array [i, j] of float k (NaN where the plane
        leaves the grid or the block). Sample volumes along the plane with sample_on_section."""
        b = self._block(block)
        r = b["krow"]
        bk = (self.section_position_mm(block, section) - b["mm_int"]) / b["mm_slope"]
        i, j = np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]), indexing="ij")
        k = (bk - r[0] * i - r[1] * j - r[3]) / r[2]
        k[(k < 0) | (k > self.shape[2] - 1)] = np.nan
        return k

    def sample_on_section(self, volume, block, section):
        """Values of a [i, j, k] volume along the oblique plane of a histology section (nearest voxel).

        Returns an [i, j] image; NaN where the plane is outside the grid. Display it with superior at the top
        (flip the i axis) and anterior on the left.
        """
        k = self.section_plane(block, section)
        valid = np.isfinite(k)
        out = np.full(k.shape, np.nan, dtype=np.float32)
        ii, jj = np.nonzero(valid)
        out[ii, jj] = np.asarray(volume, dtype=np.float32)[ii, jj, np.rint(k[valid]).astype(int)]
        return out
