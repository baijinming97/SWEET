"""
SWEET shared core: robust IO + segmentation post-processing + model selection.

Centralised here so the English and Chinese GUIs (sam_annotator_*.py) share one
implementation. All functions are dependency-light (cv2 + numpy); torch is only
touched lazily inside select_model_path().

Improvements over the original inline logic:
  - imread/imwrite that work on non-ASCII / spaced paths (the cv2.imread bug)
  - CLAHE contrast normalisation before SAM (helps low-contrast images)
  - keep ALL connected components that contain a positive point (not just largest)
  - boundary-safe STRONG negatives: small adaptive disk + drop pinched-off lobes
  - robust segmented-output filename (handles .tif/.tiff/.TIF...)
  - adaptive model selection (vit_l on GPU, vit_b on CPU)
"""
import os
import numpy as np
import cv2


# ---------------- robust image IO (fixes non-ASCII / spaced paths) ----------------
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """Drop-in for cv2.imread that survives Chinese paths / spaces / OneDrive."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except Exception:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, img):
    """Drop-in for cv2.imwrite that survives non-ASCII output paths."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        try:
            buf.tofile(path)
        except Exception:
            return False
    return ok


def segmented_path(image_path, suffix="_segmented", ext=".png"):
    """Robust replacement for image_path.replace('.tif', '_segmented.png')."""
    base, _ = os.path.splitext(image_path)
    return base + suffix + ext


# ---------------- preprocessing ----------------
def clahe_rgb(img_rgb, clip=2.0, grid=8):
    """Local contrast enhancement on the L channel; safe for grayscale-ish images."""
    if img_rgb is None or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        return img_rgb
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


# ---------------- mask post-processing ----------------
def largest_component(mask):
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m * 255
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(m)
    out[labels == largest] = 255
    return out


def label_to_color_image(label_mask):
    """Color a label mask for overlay display (ported from utils.py so the GUI
    no longer needs to import the heavy utils module)."""
    if label_mask.ndim != 2:
        raise ValueError('label_mask must be 2D array')
    labels = np.unique(label_mask)
    np.random.seed(0)
    hues = np.random.randint(0, 180, size=int(np.max(labels)) + 1)
    colors = np.zeros((len(hues), 3), dtype=np.uint8)
    colors[1:, 0] = hues[1:]
    colors[1:, 1] = 255
    colors[1:, 2] = 255
    color_mask = colors[label_mask]
    return cv2.cvtColor(color_mask, cv2.COLOR_HSV2BGR)


def keep_components_with_points(mask, points):
    """Keep every connected component containing >=1 positive point (fallback: largest)."""
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m * 255
    keep = set()
    h, w = labels.shape
    for x, y in points or []:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h and labels[yi, xi] > 0:
            keep.add(int(labels[yi, xi]))
    if not keep:
        keep.add(1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    out = np.zeros_like(m)
    for lab in keep:
        out[labels == lab] = 255
    return out


def strong_negative(mask, neg_points, pos_points, erase_ratio=0.02):
    """Boundary-safe strong red point. SAM has already pulled the boundary back
    (negatives were passed to predict); here we erase only a SMALL adaptive disk
    and then drop any lobe pinched off that contains no positive point. A click on
    the wound edge just nudges it; a thin-necked protrusion gets dropped wholesale.
    Deliberately NOT a big blunt circle (would eat the real wound at a boundary click)."""
    m = (mask > 0).astype(np.uint8)
    if not neg_points:
        return m * 255
    h, w = m.shape
    r = int(round(min(h, w) * erase_ratio))
    if r > 0:
        r = max(8, min(50, r))
        for x, y in neg_points:
            cv2.circle(m, (int(round(x)), int(round(y))), r, 0, -1)
    return keep_components_with_points(m * 255, pos_points)


def fill_holes(mask, max_frac=0.05):
    """Fill enclosed background holes (e.g. cell debris floating inside the gap) up
    to max_frac of the mask area. Debris in the wound counts as wound; a large
    deliberately-excluded interior region (or red-point cut) is left alone."""
    m = (mask > 0).astype(np.uint8)
    inv = (m == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if n <= 1:
        return m * 255
    border = (set(labels[0, :]) | set(labels[-1, :]) |
              set(labels[:, 0]) | set(labels[:, -1]))
    cap = max_frac * max(1, int(m.sum()))
    out = m.copy()
    for lab in range(1, n):
        if lab in border:                       # touches image edge => outside, not a hole
            continue
        if stats[lab, cv2.CC_STAT_AREA] <= cap:  # small enclosed debris => fill
            out[labels == lab] = 1
    return out * 255


def cleanup(mask, k_open=3, k_close=15, fill_debris=True, max_hole_frac=0.05):
    """Despeckle, bridge small edge notches from debris touching the wound edge,
    then swallow small debris fully inside the wound. Keeps the macro wound shape."""
    m = (mask > 0).astype(np.uint8)
    if k_open:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)))
    if k_close:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)))
    m = m * 255
    if fill_debris:
        m = fill_holes(m, max_hole_frac)
    return m


def finalize_mask(mask, pos_points, neg_points, fill_debris=True, close_k=15,
                  neg_ratio=0.02, do_cleanup=True):
    """Full post-processing at ORIGINAL resolution with ORIGINAL-coord points.
    Parameters are surfaced in the GUI's Advanced Settings panel."""
    out = keep_components_with_points(mask, pos_points)
    if neg_points:
        out = strong_negative(out, neg_points, pos_points, erase_ratio=neg_ratio)
    if do_cleanup:
        out = cleanup(out, k_close=close_k, fill_debris=fill_debris)
    return out


# ---------------- model selection ----------------
def select_model_path(models_dir, prefer_gpu=True):
    """Pick (path, type). vit_l when a CUDA GPU is present and the file exists,
    else vit_b. Keeps the CPU fast path on vit_b."""
    vit_b = os.path.join(models_dir, "sam_vit_b_01ec64.pth")
    vit_l = os.path.join(models_dir, "sam_vit_l_0b3195.pth")
    use_gpu = False
    if prefer_gpu:
        try:
            import torch
            use_gpu = bool(torch.cuda.is_available())
        except Exception:
            use_gpu = False
    if use_gpu and os.path.exists(vit_l):
        return vit_l, "vit_l"
    return vit_b, "vit_b"
