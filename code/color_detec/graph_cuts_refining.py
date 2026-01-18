import math
from typing import Tuple

import cv2
import numpy as np

try:
    import maxflow
except ImportError as e:
    raise ImportError("Please install PyMaxflow: pip install PyMaxflow") from e


def ged_neglogpdf(x: np.ndarray, alpha: float, beta: float, eps: float = 1e-12) -> np.ndarray:
    """
    Negative log-pdf of GED up to additive constants if desired.
    Here we keep main terms; constants help balancing but aren't strictly required.
    """
    alpha = max(float(alpha), eps)
    beta = max(float(beta), eps)

    # main exponent term
    t = (np.abs(x) / alpha) ** beta

    # include normalization constants (helps scale consistency across spaces)
    # nll = t - log(beta) + log(2*alpha*Gamma(1/beta))
    nll = t - math.log(beta) + math.log(2.0 * alpha * math.gamma(1.0 / beta))
    return nll


def compute_data_terms_from_ged(
    diff3: np.ndarray,  # (H,W,3) float32
    params_ch: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],  # (alpha,beta) per ch
    *,
    fg_bias: float = 2.0,
    bg_bias: float = 0.0,
    bg_saturate: float = 12.0,
    fg_const: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (D_bg, D_fg) each (H,W) float32.
    - D_bg: -log p(diff|background-noise GED)
    - D_fg: constant-ish + optional saturation trick
    """
    H, W, _ = diff3.shape
    d0, d1, d2 = diff3[..., 0], diff3[..., 1], diff3[..., 2]

    (a0, b0), (a1, b1), (a2, b2) = params_ch
    D_bg = (
        ged_neglogpdf(d0, a0, b0)
        + ged_neglogpdf(d1, a1, b1)
        + ged_neglogpdf(d2, a2, b2)
    ).astype(np.float32)

    # Saturation: large diffs shouldn't endlessly increase BG cost; keeps balance with smoothness
    D_bg_sat = np.minimum(D_bg, bg_saturate)

    # Foreground as weak constant prior (you can also use a "wide GED" if you want)
    D_fg = (np.full((H, W), fg_const, dtype=np.float32))

    # Add class priors / biases (optional)
    D_bg_sat = D_bg_sat + bg_bias
    D_fg = D_fg + fg_bias

    return D_bg_sat, D_fg


def graph_cut_refine_mask(
    img_bgr_ref: np.ndarray,     # reference image (for edge weights), uint8 BGR
    D_bg: np.ndarray,            # (H,W) float32
    D_fg: np.ndarray,            # (H,W) float32
    *,
    lam: float = 15.0,           # smoothness strength
    sigma: float = 15.0,         # edge sensitivity (larger => less edge effect)
    use_edge_weights: bool = True,
    neighborhood: int = 4,       # 4 or 8
) -> np.ndarray:
    """
    Binary segmentation by s-t mincut.
    Returns mask uint8 {0,255} where 255 = foreground(change).
    """
    H, W = D_bg.shape
    assert D_bg.shape == D_fg.shape

    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((H, W))

    # Unary terms: add_tedge(node, cap_source, cap_sink)
    # Convention: SOURCE=FG, SINK=BG (we choose this)
    # So: cap_source = D_fg, cap_sink = D_bg
    g.add_grid_tedges(nodeids, D_fg, D_bg)

    # Pairwise: Potts w_pq * [label_p != label_q]
    if use_edge_weights:
        gray = cv2.cvtColor(img_bgr_ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # simple gradient magnitude proxy for edges
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        # normalize-ish
        grad = grad / (np.percentile(grad, 95) + 1e-6)

        # weight per pixel: smaller across strong edges
        # w = lam * exp(-(grad^2)/(2*sigma^2))  (sigma in normalized units)
        sig = max(float(sigma), 1e-6)
        w = lam * np.exp(-(grad * grad) / (2.0 * sig * sig)).astype(np.float32)
    else:
        w = np.full((H, W), lam, dtype=np.float32)

    structure4 = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]], dtype=np.int32)
    structure8 = np.ones((3, 3), dtype=np.int32); structure8[1, 1] = 0
    structure = structure4 if neighborhood == 4 else structure8

    # add_grid_edges supports constant weights; for varying weights use add_grid_edges with weights=...
    # PyMaxflow expects weights array shaped like grid.
    g.add_grid_edges(nodeids, weights=w, structure=structure, symmetric=True)

    g.maxflow()
    seg = g.get_grid_segments(nodeids)  # True means connected to SOURCE (FG)

    mask = (seg.astype(np.uint8) * 255)
    return mask


# ---------- Usage with your existing pipeline ----------
def refine_change_mask_with_graphcuts(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    best_space: str,
    best_threshold: float,
    ged_params: Tuple["GEDParams", "GEDParams", "GEDParams"],  # from your code
    *,
    lam: float = 15.0,
) -> np.ndarray:
    # convert to chosen space
    a0 = convert_bgr(img0_bgr, best_space).astype(np.float32)
    a1 = convert_bgr(img1_bgr, best_space).astype(np.float32)
    diff3 = (a1 - a0).astype(np.float32)

    # build data terms from GED
    params_ch = tuple((p.alpha, p.beta) for p in ged_params)  # (alpha,beta) per channel
    D_bg, D_fg = compute_data_terms_from_ged(diff3, params_ch)

    # optional: use threshold mask as prior by biasing D_fg/D_bg
    # Pixels above threshold: encourage FG; below: encourage BG
    dist = np.linalg.norm(diff3.reshape(-1, 3), axis=1).reshape(diff3.shape[:2])
    prior = (dist > best_threshold).astype(np.float32)
    D_fg = D_fg - 1.5 * prior   # cheaper to be FG where threshold says change
    D_bg = D_bg - 1.5 * (1.0 - prior)

    # graph cut refinement (edge weights from reference image)
    return graph_cut_refine_mask(img0_bgr, D_bg, D_fg, lam=lam, use_edge_weights=True, neighborhood=4)


best_mask, best, ranking = detect_change_best_space(img0, img1, p_fa=1e-3)

refined = refine_change_mask_with_graphcuts(
    img0, img1,
    best_space=best.space,
    best_threshold=best.threshold,
    ged_params=best.params,
    lam=15.0,
)

cv2.imwrite("mask_threshold.png", best_mask)
cv2.imwrite("mask_graphcut.png", refined)
