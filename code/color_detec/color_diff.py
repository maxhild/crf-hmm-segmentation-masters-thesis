import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ----------------------------
# Generalized Gaussian (GED)
# p(x) = beta / (2*alpha*Gamma(1/beta)) * exp(-( |x|/alpha )^beta)
# beta=1 -> Laplace, beta=2 -> Gaussian-ish
# We fit beta via a small grid search on moment ratio, then alpha from E[|X|^p].
# ----------------------------

@dataclass(frozen=True)
class GEDParams:
    alpha: float
    beta: float


def _gamma(x: float) -> float:
    # math.gamma is fine for our parameter ranges
    return math.gamma(x)


def _ged_moment_abs(alpha: float, beta: float, p: float) -> float:
    # E[|X|^p] = alpha^p * Gamma((p+1)/beta) / Gamma(1/beta)
    return (alpha ** p) * (_gamma((p + 1.0) / beta) / _gamma(1.0 / beta))


def fit_ged_from_samples(x: np.ndarray, beta_grid: np.ndarray | None = None) -> GEDParams:
    """
    Fit GED to 1D samples x (assumed zero-mean symmetric).
    Robust-ish fit using the ratio r = E[|X|] / sqrt(E[X^2]).
    Then solve for beta by grid search; alpha by matching variance.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 100:
        # fallback to something reasonable
        return GEDParams(alpha=float(np.std(x) + 1e-6), beta=2.0)

    m1 = float(np.mean(np.abs(x)))
    m2 = float(np.mean(x * x))
    if m2 <= 1e-12:
        return GEDParams(alpha=1e-6, beta=2.0)

    r_emp = m1 / math.sqrt(m2)

    if beta_grid is None:
        # concentrate between Laplace(1) and Gaussian(2), but allow slightly wider
        beta_grid = np.concatenate([
            np.linspace(0.6, 1.0, 25, endpoint=False),
            np.linspace(1.0, 2.5, 61),
            np.linspace(2.5, 6.0, 15),
        ])

    # For unit alpha=1, compute theoretical ratio r(beta) = E|X| / sqrt(E[X^2])
    # E|X| = Gamma(2/b)/Gamma(1/b)
    # E[X^2] = Gamma(3/b)/Gamma(1/b)
    b = beta_grid
    r_th = (np.array([_gamma(2.0 / bi) for bi in b]) / np.array([_gamma(1.0 / bi) for bi in b])) / np.sqrt(
        (np.array([_gamma(3.0 / bi) for bi in b]) / np.array([_gamma(1.0 / bi) for bi in b]))
    )

    idx = int(np.argmin(np.abs(r_th - r_emp)))
    beta = float(beta_grid[idx])

    # Match variance: E[X^2] = alpha^2 * Gamma(3/beta)/Gamma(1/beta)
    c2 = _gamma(3.0 / beta) / _gamma(1.0 / beta)
    alpha = math.sqrt(m2 / max(c2, 1e-12))

    return GEDParams(alpha=float(alpha), beta=float(beta))


def sample_ged(params: GEDParams, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample from GED using inverse transform for generalized normal via
    sampling U ~ Gamma(k=1/beta, theta=alpha^beta), then X = S * U^(1/beta)
    where S is +/- with prob 1/2.
    """
    alpha, beta = params.alpha, params.beta
    k = 1.0 / beta
    theta = (alpha ** beta)
    # Gamma in numpy uses shape=k, scale=theta
    u = rng.gamma(shape=k, scale=theta, size=n)
    s = rng.choice(np.array([-1.0, 1.0]), size=n)
    return s * (u ** (1.0 / beta))


# ----------------------------
# Color spaces
# ----------------------------

_COLORSPACES: Dict[str, int] = {
    "BGR": None,  # identity
    "RGB": cv2.COLOR_BGR2RGB,
    "HSV": cv2.COLOR_BGR2HSV,
    "Lab": cv2.COLOR_BGR2Lab,
    "YCrCb": cv2.COLOR_BGR2YCrCb,
    "Luv": cv2.COLOR_BGR2Luv,
}


def convert_bgr(img_bgr: np.ndarray, space: str) -> np.ndarray:
    if space not in _COLORSPACES:
        raise ValueError(f"Unknown color space: {space}")
    code = _COLORSPACES[space]
    if code is None:
        return img_bgr
    return cv2.cvtColor(img_bgr, code)


# ----------------------------
# Change detection core
# ----------------------------

@dataclass(frozen=True)
class SpaceResult:
    space: str
    threshold: float
    params: Tuple[GEDParams, GEDParams, GEDParams]
    score: float


def _estimate_threshold_from_noise(
    params_ch: Tuple[GEDParams, GEDParams, GEDParams],
    p_fa: float,
    mc_samples: int,
    rng: np.random.Generator,
) -> float:
    # Sample noise in 3 channels, compute Euclidean distance distribution
    n = int(mc_samples)
    x0 = sample_ged(params_ch[0], n, rng)
    x1 = sample_ged(params_ch[1], n, rng)
    x2 = sample_ged(params_ch[2], n, rng)
    d = np.sqrt(x0 * x0 + x1 * x1 + x2 * x2)

    # threshold t such that P(D > t) = p_fa  => t = quantile(1 - p_fa)
    q = float(np.quantile(d, 1.0 - float(p_fa)))
    return q


def _smoothness_proxy(mask: np.ndarray) -> float:
    """
    Penalize speckle: count boundary pixels (high perimeter) relative to area.
    Lower is smoother.
    """
    m = (mask.astype(np.uint8) * 255)
    edges = cv2.Canny(m, 50, 150)
    perim = float(np.count_nonzero(edges))
    area = float(np.count_nonzero(m))
    if area <= 1.0:
        return perim  # empty mask -> ok
    return perim / area


def detect_change_best_space(
    img0_bgr: np.ndarray,
    img1_bgr: np.ndarray,
    *,
    spaces: List[str] | None = None,
    p_fa: float = 1e-3,
    noise_fit_sample_pixels: int = 200_000,
    mc_samples: int = 300_000,
    use_smoothness: bool = True,
    rng_seed: int = 0,
) -> Tuple[np.ndarray, SpaceResult, List[SpaceResult]]:
    """
    Returns:
      best_mask (H,W) uint8 in {0,255},
      best_result,
      all_results sorted by score ascending.
    """
    if spaces is None:
        spaces = ["RGB", "HSV", "Lab", "YCrCb", "Luv", "BGR"]

    if not (0.0 < p_fa < 1.0):
        raise ValueError("p_fa must be in (0,1)")

    if img0_bgr.shape != img1_bgr.shape:
        raise ValueError("Input images must have same shape")

    rng = np.random.default_rng(rng_seed)
    H, W = img0_bgr.shape[:2]
    Npix = H * W

    # subsample pixels for fitting to keep it fast
    k = min(int(noise_fit_sample_pixels), int(Npix))
    idx = rng.choice(Npix, size=k, replace=False)

    results: List[SpaceResult] = []

    for sp in spaces:
        a0 = convert_bgr(img0_bgr, sp).astype(np.float32)
        a1 = convert_bgr(img1_bgr, sp).astype(np.float32)

        # channel differences
        diff = (a1 - a0).reshape(-1, 3)

        # fit noise on *all* differences (in practice you might want robust trimming)
        # robust trimming: remove extreme tail that likely contains true changes
        dnorm = np.linalg.norm(diff[idx], axis=1)
        trim_q = float(np.quantile(dnorm, 0.90))
        keep = idx[dnorm <= trim_q]
        if keep.size < 200:
            keep = idx

        ch0 = diff[keep, 0]
        ch1 = diff[keep, 1]
        ch2 = diff[keep, 2]

        p0 = fit_ged_from_samples(ch0)
        p1 = fit_ged_from_samples(ch1)
        p2 = fit_ged_from_samples(ch2)

        thr = _estimate_threshold_from_noise((p0, p1, p2), p_fa=p_fa, mc_samples=mc_samples, rng=rng)

        # apply threshold
        dist = np.linalg.norm(diff, axis=1).reshape(H, W)
        mask = (dist > thr).astype(np.uint8)

        # Score: expected FP is fixed to p_fa * Npix by construction,
        # so we add a proxy for "implausible" detection: prefer smaller masks
        # and smoother masks (optional).
        area = float(mask.sum()) / float(Npix)  # fraction changed
        smooth = _smoothness_proxy(mask) if use_smoothness else 0.0
        score = 0.7 * area + (0.3 * smooth if use_smoothness else 0.0)

        results.append(SpaceResult(space=sp, threshold=float(thr), params=(p0, p1, p2), score=float(score)))

    results.sort(key=lambda r: r.score)
    best = results[0]

    # recompute best mask at full res
    b0 = convert_bgr(img0_bgr, best.space).astype(np.float32)
    b1 = convert_bgr(img1_bgr, best.space).astype(np.float32)
    bdiff = (b1 - b0).reshape(-1, 3)
    bdist = np.linalg.norm(bdiff, axis=1).reshape(H, W)
    best_mask = (bdist > best.threshold).astype(np.uint8) * 255

    return best_mask, best, results


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    img0 = cv2.imread("frame_000.png")  # BGR
    img1 = cv2.imread("frame_001.png")
    mask, best, ranking = detect_change_best_space(img0, img1, p_fa=1e-3)
    
    refined = refine_change_mask_with_graphcuts(
    img0, img1,
    best_space=best.space,
    best_threshold=best.threshold,
    ged_params=best.params,
    lam=15.0,
    )

    cv2.imwrite("mask_threshold.png", best_mask)
    cv2.imwrite("mask_graphcut.png", refined)


    print("Best space:", best.space, "threshold:", best.threshold, "score:", best.score)
    cv2.imwrite("change_mask.png", mask)
