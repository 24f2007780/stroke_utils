"""display_utils - plotting helpers shared by the C-Stroke notebooks.

    set_style            matplotlib defaults (light surface, recessive axes, readable type)
    MARKER_COLORS        one fixed colour per IHC marker (and "HE" for the infarct)
    GRADE_COLORS         ordinal ramp for annotation grades (0 = not annotated)
    show_image           image on an axes with an optional scale bar and title
    draw_annotation      outline (or fill) a SectionAnnotation on an image of a slide_utils.Region
    scale_bar            physical scale bar for an image of a slide_utils.Region
    show_refspace        one slice of a reference-space volume, oriented and scaled in millimetres
    show_mesh            a lit triangle surface on a 3D axes
"""

import numpy as np

__all__ = ["MARKER_COLORS", "GRADE_COLORS", "STAIN_COLORS", "INFARCT_COLOR", "set_style", "show_image", "draw_annotation",
           "scale_bar", "show_refspace", "show_mesh"]

# Categorical colours in a fixed order (validated for colour-vision deficiency); a marker keeps its colour.
MARKER_COLORS = {"GFAP": "#2a78d6", "CD68": "#eb6834", "CD34": "#1baf7a", "HIF1a": "#eda100",
                 "FIB": "#e87ba4", "APP": "#008300", "GAP43": "#4a3aa7", "HE": "#e34948", "AQP4": "#52514e"}
GRADE_COLORS = {0: "#e6e5e0", 1: "#86b6ef", 2: "#2a78d6", 3: "#104281"}
STAIN_COLORS = {"IHC": "#b5651d", "N": "#4a5fc1", "HE": "#d16ba5", "M": "#5a3b1f"}
INFARCT_COLOR = "#e34948"


def set_style():
    """Matplotlib defaults for the C-Stroke notebooks."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 150, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#b5b4ae", "axes.labelcolor": "#2b2b29", "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 10, "image.interpolation": "nearest",
    })


def scale_bar(ax, region, length_mm, loc="lower right", color="black"):
    """Add a scale bar of `length_mm` to an axes showing an image of `region`."""
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    px = length_mm * 1000.0 / region.mpp
    label = f"{length_mm:g} mm" if length_mm >= 1 else f"{length_mm * 1000:g} µm"
    ax.add_artist(AnchoredSizeBar(ax.transData, px, label, loc, pad=0.4, color=color, frameon=False,
                                  size_vertical=max(px / 40, 1)))


def show_image(ax, image, region=None, title=None, bar_mm=None, axis_off=True):
    """Display an RGB image; with `region` and `bar_mm`, add a scale bar of that length."""
    ax.imshow(image)
    if title:
        ax.set_title(title)
    if region is not None and bar_mm:
        scale_bar(ax, region, bar_mm)
    if axis_off:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)


def draw_annotation(ax, annotation, region, colors=None, linewidth=0.9, fill_alpha=0.0, clip=True):
    """Outline the polygons of a SectionAnnotation on an axes showing an image of `region`.

    colors      dict grade -> colour; defaults to GRADE_COLORS (single-class markers use the darkest step)
    fill_alpha  > 0 also fills the polygons with that opacity
    clip        keep the axes limits of the image (polygons outside it are cut off)
    """
    from matplotlib.patches import Polygon as MplPolygon

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if colors is None:
        single = len(annotation.classes) <= 1
        colors = {g: (GRADE_COLORS[3] if single else GRADE_COLORS.get(g, GRADE_COLORS[3])) for g in range(1, 4)}
    for r in annotation.records:
        color = colors.get(r["grade"], GRADE_COLORS[3])
        geoms = r["geometry"].geoms if hasattr(r["geometry"], "geoms") else [r["geometry"]]
        for poly in geoms:
            if poly.geom_type != "Polygon":
                continue
            c, rr = region.to_pixels(*poly.exterior.xy)
            ax.plot(c, rr, color=color, lw=linewidth)
            if fill_alpha:
                ax.add_patch(MplPolygon(np.column_stack([c, rr]), closed=True, facecolor=color, alpha=fill_alpha,
                                        edgecolor="none"))
    if clip:
        ax.set_xlim(xlim); ax.set_ylim(ylim)


def show_refspace(ax, volume, plane, index, spacing, cmap="gray", vmin=None, vmax=None, alpha=1.0, title=None):
    """Display one slice of a reference-space volume ([i, j, k]; i inferior->superior, j anterior->posterior,
    k left->right) in anatomical orientation (neurological convention) with millimetre aspect.

    plane   "sagittal" (fixed k; anterior left, superior up), "coronal" (fixed j; subject's left on the left,
            superior up) or "axial" (fixed i; subject's left on the left, anterior up)
    """
    si, sj, sk = spacing
    if plane == "sagittal":
        img, aspect = volume[::-1, :, index], si / sj
    elif plane == "coronal":
        img, aspect = volume[::-1, index, :], si / sk
    elif plane == "axial":
        img, aspect = volume[index, :, :], sj / sk
    else:
        raise ValueError("plane must be 'sagittal', 'coronal' or 'axial'")
    handle = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect, alpha=alpha)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    return handle


def show_mesh(ax, vertices, faces, color=INFARCT_COLOR, light=(0.4, -0.6, 0.7), elev=20, azim=-60):
    """Draw a triangle mesh on a 3D axes with simple directional lighting and equal axis scaling.

    vertices  (n, 3) array of x, y, z coordinates; faces  (m, 3) vertex indices
    """
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    l = np.asarray(light, dtype=float) / np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.abs(normals @ l)
    colors = np.clip(np.outer(shade, to_rgb(color)), 0, 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=colors, edgecolor="none", linewidth=0))
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=elev, azim=azim)
