"""
Auto Retouch engine
-------------------
1) Face detection + landmarks (MediaPipe) -> face skin polygon, minus eyes / brows / lips
2) Person segmentation (MediaPipe Selfie Segmentation, pre-trained, Apache-2.0)
   + a skin colour model learned from the detected face -> neck / body / hands skin
3) Dodge & burn: smooth, bounded corrections (brightness + colour) that even out
   mid-scale unevenness while leaving the pore-level texture untouched.
"""
from dataclasses import dataclass, field

import os
import shutil
import tempfile

import cv2
import numpy as np
import mediapipe as mp


def _ascii_mediapipe_fix(force=False):
    """MediaPipe's C++ loader cannot open model files under folders with non-ASCII names
    (e.g. Persian). In that case copy the few models we need to an ASCII folder and
    point MediaPipe there."""
    from mediapipe.python import solution_base as sb
    mod = os.path.abspath(sb.__file__)
    root = os.path.dirname(os.path.dirname(os.path.dirname(mod)))
    if root.isascii() and not force:
        return None
    names = ("face_detection", "face_landmark", "selfie_segmentation", "iris_landmark")
    cands = [os.environ.get("PROGRAMDATA"), r"C:\Users\Public", tempfile.gettempdir(), "C:\\"]
    for base in cands:
        if not base or not base.isascii():
            continue
        dst = os.path.join(base, "AutoRetouch_mp")
        try:
            for n in names:
                src = os.path.join(root, "mediapipe", "modules", n)
                tgt = os.path.join(dst, "mediapipe", "modules", n)
                if os.path.isdir(src) and not os.path.isdir(tgt):
                    shutil.copytree(src, tgt, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            sb.__file__ = os.path.join(dst, "mediapipe", "python", "solution_base.py")
            return dst
        except Exception:
            continue
    return None


try:
    _ascii_mediapipe_fix()
except Exception:
    pass

_FD = mp.solutions.face_detection
_FM = mp.solutions.face_mesh
_SS = mp.solutions.selfie_segmentation

MODES = ("auto", "face", "half", "full")

# Tuning constants of the multi-scale frequency separation (relative to face size in px)
TEX_SCALE = 1.0 / 140.0   # pore / micro-texture cut-off
MID_SCALE = 1.0 / 20.0    # uneven-lighting band cut-off
TGT_SCALE = 1.0 / 5.0     # averaging window for the mid band
EDGE_L = 6.0              # bilateral edge sensitivity on L  (0..100)
EDGE_AB = 3.5             # bilateral edge sensitivity on a/b
CLIP_L = 7.0              # max L correction (tanh knee)
CLIP_AB = 3.0             # max a/b correction (tanh knee)
MASK_LIGHT_RANGE = 16.0   # L units reached by the "mask light" slider at +-100


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def resize_max(img, m, interp=cv2.INTER_AREA):
    h, w = img.shape[:2]
    s = m / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=interp)


def gblur(a, sigma):
    """Gaussian blur; very large sigmas are done on a down-scaled copy (fast)."""
    if sigma < 6:
        return cv2.GaussianBlur(a, (0, 0), max(sigma, 0.1))
    f = max(2, int(sigma / 3))
    h, w = a.shape[:2]
    small = cv2.resize(a, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def nblur(x, w, sigma):
    """Normalised (mask-weighted) blur: average of x over the masked area only."""
    num = gblur(x * w, sigma)
    den = gblur(w, sigma)
    return num / (den + 1e-4)


def eblur(x, sigma, edge_sigma):
    """Edge-preserving blur (bilateral). Large sigmas on a down-scaled copy."""
    x = np.asarray(x, np.float32)
    sigma = max(float(sigma), 0.5)
    if sigma < 6:
        return cv2.bilateralFilter(x, 0, float(edge_sigma), sigma)
    f = max(2, int(sigma / 3))
    h, w = x.shape[:2]
    small = cv2.resize(x, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    small = cv2.bilateralFilter(small, 0, float(edge_sigma), sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def nblur_ep(x, w, sigma, edge_sigma):
    """Edge-preserving, mask-weighted local average of x."""
    num = eblur(x * w, sigma, edge_sigma)
    den = gblur(w, sigma)
    return num / (den + 1e-4)


def _hull(pts):
    return cv2.convexHull(np.asarray(pts, np.float32).reshape(-1, 1, 2)).astype(np.int32)


def _idx(conn):
    return sorted({i for c in conn for i in c})


@dataclass
class Face:
    box: tuple
    oval: np.ndarray
    excl: list
    face_px: float
    center: tuple
    mesh_ok: bool


@dataclass
class Analysis:
    shape: tuple
    bgr: np.ndarray = None
    faces: list = field(default_factory=list)
    person: np.ndarray = None
    lab_small: np.ndarray = None
    person_small: np.ndarray = None
    med_face_px: float = 0.0


# ----------------------------------------------------------------------------
# detection
# ----------------------------------------------------------------------------
def _iou(a, b):
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0


def _nms(boxes, thr=0.35):
    boxes = sorted(boxes, key=lambda b: -b[4])
    keep = []
    for b in boxes:
        if all(_iou(b, k) < thr for k in keep):
            keep.append(b)
    return keep


def _detect_boxes(rgb):
    H, W = rgb.shape[:2]
    boxes = []

    def run(det, im, ox, oy):
        res = det.process(np.ascontiguousarray(im))
        h_, w_ = im.shape[:2]
        for d in (res.detections or []):
            bb = d.location_data.relative_bounding_box
            boxes.append((ox + bb.xmin * w_, oy + bb.ymin * h_, bb.width * w_, bb.height * h_, float(d.score[0])))

    with _FD.FaceDetection(model_selection=1, min_detection_confidence=0.4) as far, \
            _FD.FaceDetection(model_selection=0, min_detection_confidence=0.5) as near:
        run(far, rgb, 0, 0)
        run(near, rgb, 0, 0)
        boxes = _nms(boxes)
        if not boxes:  # small faces: look at overlapping tiles
            tw, th = int(W * 0.62), int(H * 0.62)
            for oy in (0, H - th):
                for ox in (0, W - tw):
                    run(far, rgb[oy:oy + th, ox:ox + tw], ox, oy)
            boxes = _nms(boxes)
    boxes = [b for b in boxes if b[2] > 12 and b[3] > 12 and b[4] > 0.3]
    return boxes[:12]


def _mesh_points(fm, rgb, box):
    x, y, w, h, _ = box
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * 2.2
    S = 640
    sc = S / side
    M = np.array([[sc, 0, S / 2 - cx * sc], [0, sc, S / 2 - cy * sc]], np.float32)
    crop = cv2.warpAffine(rgb, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    res = fm.process(crop)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([[l.x * S, l.y * S] for l in lm], np.float32)
    pts[:, 0] = (pts[:, 0] - M[0, 2]) / sc
    pts[:, 1] = (pts[:, 1] - M[1, 2]) / sc
    mc = pts.mean(0)
    if abs(mc[0] - cx) > w * 0.6 or abs(mc[1] - cy) > h * 0.6:
        return None  # locked onto a different face
    return pts


def _ellipse_face(box):
    x, y, w, h, _ = box
    cx, cy = x + w / 2, y + h / 2
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    oval = np.stack([cx + 0.52 * w * np.cos(t), cy + 0.58 * h * np.sin(t)], 1)
    excl = []
    for ex in (0.30, 0.70):  # rough eye / brow zones
        c = (x + ex * w, y + 0.40 * h)
        e = np.stack([c[0] + 0.15 * w * np.cos(t), c[1] + 0.07 * h * np.sin(t)], 1)
        excl.append(_hull(e))
        c2 = (x + ex * w, y + 0.31 * h)
        e2 = np.stack([c2[0] + 0.17 * w * np.cos(t), c2[1] + 0.05 * h * np.sin(t)], 1)
        excl.append(_hull(e2))
    c = (cx, y + 0.78 * h)
    lips = np.stack([c[0] + 0.22 * w * np.cos(t), c[1] + 0.08 * h * np.sin(t)], 1)
    excl.append(_hull(lips))
    return _hull(oval), excl


def analyze(bgr):
    """Detect faces + person once. Returns an Analysis (mode independent)."""
    H, W = bgr.shape[:2]
    an = Analysis(shape=(H, W), bgr=bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    boxes = _detect_boxes(rgb)
    ov, le, re_ = _idx(_FM.FACEMESH_FACE_OVAL), _idx(_FM.FACEMESH_LEFT_EYE), _idx(_FM.FACEMESH_RIGHT_EYE)
    lb, rb, lips = _idx(_FM.FACEMESH_LEFT_EYEBROW), _idx(_FM.FACEMESH_RIGHT_EYEBROW), _idx(_FM.FACEMESH_LIPS)

    if boxes:
        with _FM.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                          min_detection_confidence=0.3) as fm:
            for box in boxes:
                pts = _mesh_points(fm, rgb, box)
                if pts is not None:
                    oval = _hull(pts[ov])
                    excl = [_hull(pts[i]) for i in (le, re_, lb, rb, lips)]
                    ok = True
                else:
                    oval, excl = _ellipse_face(box)
                    ok = False
                bx, by, bw, bh = cv2.boundingRect(oval)
                an.faces.append(Face(box, oval, excl, float(bh), (bx + bw / 2, by + bh / 2), ok))
        an.med_face_px = float(np.median([f.face_px for f in an.faces]))

    try:
        with _SS.SelfieSegmentation(model_selection=0) as ss:
            an.person = ss.process(rgb).segmentation_mask.astype(np.float32)
    except Exception:
        an.person = None
    return an


# ----------------------------------------------------------------------------
# mask
# ----------------------------------------------------------------------------
def resolve_mode(an, mode):
    if mode != "auto":
        return mode
    if not an.faces:
        return "face"
    r = an.med_face_px / an.shape[0]
    if r > 0.45:
        return "face"
    if r > 0.15:
        return "half"
    return "full"


def build_mask(an, mode="auto"):
    """Skin mask (float32 0..1) at the analysis resolution + info dict."""
    H, W = an.shape
    used = resolve_mode(an, mode)
    info = {"mode": used, "faces": len(an.faces), "mesh": sum(f.mesh_ok for f in an.faces)}
    if not an.faces:
        return np.zeros((H, W), np.float32), info

    face_poly = np.zeros((H, W), np.uint8)
    excl = np.zeros((H, W), np.uint8)
    for f in an.faces:
        cv2.fillConvexPoly(face_poly, f.oval, 255)
        for e in f.excl:
            cv2.fillConvexPoly(excl, e, 255)
    fp = an.med_face_px
    k = max(1, int(fp * 0.014)) | 1
    excl = cv2.dilate(excl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1)))
    ke = max(1, int(fp * 0.02))
    face_poly = cv2.erode(face_poly, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ke * 2 + 1, ke * 2 + 1)))

    ext = np.zeros((H, W), np.uint8)
    if used in ("half", "full"):
        ext = _skin_extension(an, face_poly, excl, used)

    mask = np.maximum(face_poly, ext)
    mask[excl > 0] = 0
    return mask.astype(np.float32) / 255.0, info


def _skin_extension(an, face_poly, excl, used):
    H, W = an.shape
    s = 1000.0 / max(H, W)
    s = min(1.0, s)
    sw, sh = max(1, int(W * s)), max(1, int(H * s))
    small = cv2.resize(an.bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    fpoly = cv2.resize(face_poly, (sw, sh), interpolation=cv2.INTER_NEAREST)
    ex = cv2.resize(excl, (sw, sh), interpolation=cv2.INTER_NEAREST)

    if an.person is not None:
        person = cv2.resize(an.person, (sw, sh), interpolation=cv2.INTER_LINEAR)
        kk = max(3, int(0.03 * max(sw, sh))) | 1
        person_ok = cv2.dilate((person > 0.35).astype(np.uint8), np.ones((kk, kk), np.uint8)) > 0
    else:
        person_ok = np.ones((sh, sw), bool)

    sel = (cv2.erode(fpoly, np.ones((5, 5), np.uint8)) > 0) & (ex == 0)
    if sel.sum() < 150:
        return np.zeros((H, W), np.uint8)

    ab = lab[..., 1:3]
    Ls = lab[..., 0][sel]
    a_b = ab[sel]
    mu = a_b.mean(0)
    cov = np.cov(a_b.T) + np.eye(2, dtype=np.float32) * 1.0
    inv = np.linalg.inv(cov).astype(np.float32)
    d = ab - mu.astype(np.float32)
    m2 = d[..., 0] ** 2 * inv[0, 0] + 2 * d[..., 0] * d[..., 1] * inv[0, 1] + d[..., 1] ** 2 * inv[1, 1]
    lmin = np.percentile(Ls, 3) - 24
    lmax = np.percentile(Ls, 99.7) + 12
    skin = (m2 < 9.0) & (lab[..., 0] > lmin) & (lab[..., 0] < lmax) & person_ok
    skin = skin.astype(np.uint8) * 255
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    fp_s = an.med_face_px * s
    seed = cv2.dilate(fpoly, np.ones((int(fp_s * 0.2) | 1, int(fp_s * 0.2) | 1), np.uint8)) > 0
    # hair / head zone: never extend the skin colour model over the top half of the head
    hair = np.zeros((sh, sw), np.uint8)
    for f in an.faces:
        cx, cy = f.center[0] * s, f.center[1] * s
        fps = f.face_px * s
        x0, x1 = int(cx - 1.1 * fps), int(cx + 1.1 * fps)
        y1 = int(cy + 0.12 * fps)
        hair[max(0, 0):max(0, y1), max(0, x0):max(0, x1)] = 255
    skin[hair > 0] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
    out = np.zeros((sh, sw), np.uint8)
    min_area = (fp_s * 0.12) ** 2
    if used == "half":
        ys = np.arange(sh)[:, None]
        cy = np.mean([f.center[1] for f in an.faces]) * s
        vert = (ys < cy + 2.4 * fp_s)
    for i in range(1, n):
        comp = labels == i
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        if used == "half":
            if (comp & seed).any():
                out[comp & vert] = 255
        else:
            out[comp] = 255
    out = cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)
    return (out > 127).astype(np.uint8) * 255


def feather(mask, px=20, long_side=2000):
    """Soft mask edge. px = fade width in pixels for a 2000 px image (scaled with image size)."""
    sigma = max(0.5, 0.4 * float(px) * float(long_side) / 2000.0)
    return np.clip(cv2.GaussianBlur(mask, (0, 0), sigma), 0, 1)


# ----------------------------------------------------------------------------
# retouch - multi-scale frequency separation (texture & volume safe)
# ----------------------------------------------------------------------------
def compute_delta(lab, m, face_px):
    """
    L is split into three bands (scales derive from face_px):
        fine   = L - blur(L, s_tex)               pores, lashes  -> never touched
        mid    = blur(L, s_tex) - blur(L, s_mid)  blotches, under-eye, redness, shine -> evened out
        volume = blur(L, s_mid)                   bone structure, broad shading -> never touched
    The mid band is pulled toward an EDGE-PRESERVING masked local average, so real shading
    survives while isolated unevenness is levelled. The returned delta contains no fine and
    no coarse frequencies, so adding it can neither soften pores nor flatten volume.
    lab: float32 Lab (L 0..100). m: soft skin mask 0..1. Returns (dL, da, db).
    """
    L, a, b = cv2.split(lab)
    w = np.clip(m, 0, 1).astype(np.float32)
    s_tex = max(1.0, face_px * TEX_SCALE)
    s_mid = max(2.5, face_px * MID_SCALE)
    s_tgt = max(5.0, face_px * TGT_SCALE)

    mid = gblur(L, s_tex) - gblur(L, s_mid)
    tgt = nblur_ep(mid, w, s_tgt, EDGE_L)
    d = tgt - mid
    dL = CLIP_L * np.tanh(d / CLIP_L)
    dL = np.where(dL < 0, dL * 0.8, dL)  # lift shadows slightly more than darken

    out = [dL]
    for ch in (a, b):
        s_tex_c = s_tex * 1.6
        s_mid_c = max(4.0, face_px / 15.0)
        s_tgt_c = max(8.0, face_px / 3.5)
        midc = gblur(ch, s_tex_c) - gblur(ch, s_mid_c)
        dc = nblur_ep(midc, w, s_tgt_c, EDGE_AB) - midc
        out.append(0.8 * CLIP_AB * np.tanh(dc / CLIP_AB))
    return tuple(x.astype(np.float32) for x in out)


ADJ_KEYS = ("brightness", "contrast", "highlights", "shadows", "whites", "blacks", "sharpen", "blur",
            "temp", "tint", "sat", "vib", "hue", "red", "green", "blue")


def adj_active(p):
    return any(abs(float(p.get(k, 0.0))) > 1e-6 for k in ADJ_KEYS)


def _smooth01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def apply_maps(bgr_u8, maps, p, db=None, s=(1.0, 1.0, 0.0)):
    """
    maps: float32 HxWx4 = (mask, dL, da, db) at the SAME size as bgr_u8.
    p: dict of parameters. Skin retouch: blend, strength, warm, texture, mask_light (-1..1).
       Whole-image adjustments (all -1..1, blur 0..1): see ADJ_KEYS. Flags: db_limit, adj_limit.
    db: manual dodge/burn map in L units (same size) or None.
    s: (sigma_texture, sigma_sharpen, sigma_blur) in pixels of THIS array.
    """
    s_tex, s_sh, s_bl = s
    m = maps[..., 0]
    wgt = m * float(p.get("blend", 1.0))
    strength = float(p.get("strength", 1.0))
    adj = adj_active(p)
    G = m if p.get("adj_limit") else 1.0                  # weight for whole-image adjustments
    G3 = m[..., None] if p.get("adj_limit") else 1.0

    lab = cv2.cvtColor(bgr_u8.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].copy()
    corr = strength * maps[..., 1]
    tex = float(p.get("texture", 1.0))
    if abs(tex - 1.0) > 1e-3:
        corr = corr + (tex - 1.0) * (L - cv2.GaussianBlur(L, (0, 0), max(0.5, s_tex)))
    L1 = L + wgt * corr
    act = wgt > 1e-3

    ml = float(p.get("mask_light", 0.0))
    if abs(ml) > 1e-6:                                    # dodge / burn under the mask
        L1 = L1 + ml * MASK_LIGHT_RANGE * m
        act = act | (m > 1e-3)
    if db is not None:                                    # manual dodge / burn brush
        dbv = db * (m if p.get("db_limit") else 1.0)
        L1 = L1 + dbv
        act = act | (np.abs(dbv) > 1e-3)

    # ---- whole-image tone ----
    br = float(p.get("brightness", 0.0))
    if br:
        x = np.clip(L1 / 100.0, 0.0, 1.0)
        L1 = L1 + (100.0 * np.power(x, 2.0 ** (-br * 1.2)) - L1) * G
    hl, sh = float(p.get("highlights", 0.0)), float(p.get("shadows", 0.0))
    if hl or sh:
        x = np.clip(L1 / 100.0, 0.0, 1.0)
        L1 = L1 + (hl * 28.0 * _smooth01((x - 0.5) * 2.0) + sh * 28.0 * _smooth01(1.0 - x * 2.0)) * G
    wt, bk = float(p.get("whites", 0.0)), float(p.get("blacks", 0.0))
    if wt or bk:
        x = np.clip(L1 / 100.0, 0.0, 1.0)
        L1 = L1 + (wt * 22.0 * _smooth01((x - 0.7) / 0.3) + bk * 22.0 * _smooth01((0.3 - x) / 0.3)) * G
    ct = float(p.get("contrast", 0.0))
    if ct:
        x = np.clip(L1 / 100.0, 0.0, 1.0)
        L1 = L1 + ((x + ct * (_smooth01(x) - x)) * 100.0 - L1) * G
    sp = float(p.get("sharpen", 0.0))
    if sp:
        amt = sp * 1.5 if sp > 0 else sp * 1.0
        L1 = L1 + amt * (L1 - cv2.GaussianBlur(L1, (0, 0), max(0.5, s_sh))) * G
    lab[..., 0] = np.clip(L1, 0.0, 100.0)

    # ---- chroma ----
    a = lab[..., 1] + wgt * (strength * maps[..., 2] + float(p.get("warm", 0.0)) * 0.04)
    b = lab[..., 2] + wgt * (strength * maps[..., 3] + float(p.get("warm", 0.0)) * 0.16)
    tp, tn = float(p.get("temp", 0.0)), float(p.get("tint", 0.0))
    if tp or tn:
        b = b + tp * 14.0 * G
        a = a + (tp * 3.0 + tn * 12.0) * G
    hu, sa, vi = float(p.get("hue", 0.0)), float(p.get("sat", 0.0)), float(p.get("vib", 0.0))
    if hu or sa or vi:
        a0, b0 = a, b
        if hu:
            th = hu * np.pi
            a0, b0 = a * np.cos(th) - b * np.sin(th), a * np.sin(th) + b * np.cos(th)
        f = 1.0 + sa
        if vi:
            c = np.sqrt(a0 * a0 + b0 * b0)
            f = f * (1.0 + vi * (1.0 - np.clip(c / 60.0, 0.0, 1.0)))
        a0, b0 = a0 * f, b0 * f
        a, b = a + (a0 - a) * G, b + (b0 - b) * G
    lab[..., 1], lab[..., 2] = a, b

    new = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), 0.0, 1.0)

    # ---- RGB channel gains, blur ----
    for idx, key in ((2, "red"), (1, "green"), (0, "blue")):
        v = float(p.get(key, 0.0))
        if v:
            new[..., idx] = np.clip(new[..., idx] * (1.0 + v * 0.5 * G), 0.0, 1.0)
    bl = float(p.get("blur", 0.0))
    if bl > 0 and s_bl > 0.3:
        new = new + (cv2.GaussianBlur(new, (0, 0), s_bl) - new) * G3

    if adj:
        act = (act | (m > 1e-3)) if p.get("adj_limit") else np.ones_like(act)
    new = (new * 255.0 + 0.5).astype(np.uint8)
    return np.where(act[..., None], new, bgr_u8)


def render_full(bgr, maps_p, p, db_p=None, s_p=(1.0, 1.0, 0.0), progress=None):
    """Apply preview-resolution maps to the full-resolution image, strip by strip.
    Texture / sharpen / blur filters are computed at FULL resolution (with row padding)."""
    H, W = bgr.shape[:2]
    Hp, Wp = maps_p.shape[:2]
    k = H / Hp
    s = tuple(v * k for v in s_p)
    pad = int(3 * max(s)) + 2
    out = np.empty_like(bgr)
    step = 512
    xs = ((np.arange(W, dtype=np.float32) + 0.5) * (Wp / W) - 0.5)
    for y0 in range(0, H, step):
        y1 = min(H, y0 + step)
        a0, a1 = max(0, y0 - pad), min(H, y1 + pad)
        ys = ((np.arange(a0, a1, dtype=np.float32) + 0.5) * (Hp / H) - 0.5)
        mx, my = np.meshgrid(xs, ys)
        up = cv2.remap(maps_p, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dbu = None
        if db_p is not None:
            dbu = cv2.remap(db_p, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        res = apply_maps(bgr[a0:a1], up, p, db=dbu, s=s)
        out[y0:y1] = res[y0 - a0:y0 - a0 + (y1 - y0)]
        if progress:
            progress(y1 / H)
    return out
