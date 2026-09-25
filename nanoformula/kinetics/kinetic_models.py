"""
Mathematical models and parameter fitting for pharmaceutical drug release kinetics.
Implements Zero-Order, First-Order, Higuchi, Korsmeyer-Peppas, and Hixson-Crowell equations.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def zero_order(t: np.ndarray, k0: float) -> np.ndarray:
    """Zero-order release: Q(t) = k0 * t"""
    return np.clip(k0 * t, 0.0, 100.0)


def first_order(t: np.ndarray, k1: float) -> np.ndarray:
    """First-order release: Q(t) = 100 * (1 - exp(-k1 * t))"""
    return 100.0 * (1.0 - np.exp(-k1 * t))


def higuchi(t: np.ndarray, kh: float) -> np.ndarray:
    """Higuchi matrix diffusion model: Q(t) = kh * sqrt(t)"""
    return np.clip(kh * np.sqrt(t), 0.0, 100.0)


def korsmeyer_peppas(t: np.ndarray, k_kp: float, n: float) -> np.ndarray:
    """Korsmeyer-Peppas power law: Q(t) = k_kp * (t ** n)"""
    return np.clip(k_kp * (t ** n), 0.0, 100.0)


def hixson_crowell(t: np.ndarray, khc: float) -> np.ndarray:
    """Hixson-Crowell cube root dissolution model: Q(t) = 100 * (1 - (1 - khc * t)^3)"""
    val = np.maximum(0.0, 1.0 - khc * t)
    return np.clip(100.0 * (1.0 - val**3), 0.0, 100.0)


def classify_release_mechanism(n_exponent: float) -> Dict[str, str]:
    """
    Classifies drug release mechanism for spherical nanoparticulate matrices
    based on the Korsmeyer-Peppas diffusion exponent (n) (Peppas, 1985).
    """
    if n_exponent <= 0.43:
        mechanism = "Fickian Diffusion (Case I Transport)"
        description = "Drug release is governed strictly by classical molecular diffusion through the polymer matrix. Polymer chain relaxation rate is significantly faster than diffusion rate."
        rate_limiting = "Concentration-gradient driven diffusion through water-filled matrix pores."
    elif 0.43 < n_exponent < 0.85:
        mechanism = "Anomalous (Non-Fickian) Transport"
        description = "Coupled transport mechanism involving both Fickian diffusion and time-dependent polymer chain relaxation / swelling."
        rate_limiting = "Combined matrix diffusion and hydrolytic ester bond relaxation."
    elif 0.85 <= n_exponent < 1.0:
        mechanism = "Case II Transport (Relaxation / Erosion Controlled)"
        description = "Zero-order like drug release governed by polymer bulk degradation, surface erosion, and matrix dissolution."
        rate_limiting = "Polymer matrix hydrolytic degradation and cleavage."
    else:
        mechanism = "Super Case II Transport"
        description = "Accelerated release driven by severe osmotic stress, polymer swelling, and rapid matrix breakdown."
        rate_limiting = "Rapid bulk matrix structural disintegration."

    return {
        "mechanism": mechanism,
        "description": description,
        "rate_limiting_step": rate_limiting
    }


def fit_all_kinetic_models(time_points: np.ndarray, cumulative_release: np.ndarray) -> Dict[str, Any]:
    """
    Fits empirical release curves to all 5 standard biopharmaceutics kinetic equations.
    Calculates rate constants, R² goodness-of-fit, and determines the dominant release mechanism.
    """
    t = np.array(time_points, dtype=float)
    y = np.array(cumulative_release, dtype=float)

    # Avoid zero or negative time
    valid_mask = t > 0
    t_v = t[valid_mask]
    y_v = y[valid_mask]

    results = {}

    # 1. Zero-Order Fit
    try:
        popt, _ = curve_fit(zero_order, t_v, y_v, p0=[1.0], bounds=(0, 50))
        y_pred = zero_order(t_v, *popt)
        r2_zero = max(0.0, float(r2_score(y_v, y_pred)))
        results["Zero-Order"] = {
            "rate_constant_k0": round(float(popt[0]), 4),
            "unit": "% / hour",
            "R2": round(r2_zero, 4),
            "formula": "Q(t) = k0 * t"
        }
    except Exception:
        results["Zero-Order"] = {"rate_constant_k0": 0.0, "R2": 0.0, "unit": "% / hour"}

    # 2. First-Order Fit
    try:
        popt, _ = curve_fit(first_order, t_v, y_v, p0=[0.05], bounds=(0, 5))
        y_pred = first_order(t_v, *popt)
        r2_first = max(0.0, float(r2_score(y_v, y_pred)))
        results["First-Order"] = {
            "rate_constant_k1": round(float(popt[0]), 4),
            "unit": "hour^-1",
            "R2": round(r2_first, 4),
            "formula": "Q(t) = 100 * (1 - exp(-k1 * t))"
        }
    except Exception:
        results["First-Order"] = {"rate_constant_k1": 0.0, "R2": 0.0, "unit": "hour^-1"}

    # 3. Higuchi Model Fit
    try:
        popt, _ = curve_fit(higuchi, t_v, y_v, p0=[10.0], bounds=(0, 100))
        y_pred = higuchi(t_v, *popt)
        r2_higuchi = max(0.0, float(r2_score(y_v, y_pred)))
        results["Higuchi"] = {
            "rate_constant_kh": round(float(popt[0]), 4),
            "unit": "% / hour^0.5",
            "R2": round(r2_higuchi, 4),
            "formula": "Q(t) = kh * t^0.5"
        }
    except Exception:
        results["Higuchi"] = {"rate_constant_kh": 0.0, "R2": 0.0, "unit": "% / hour^0.5"}

    # 4. Korsmeyer-Peppas Fit (evaluated for Q <= 60% as per classical polymer theory)
    try:
        kp_mask = (t_v > 0) & (y_v <= 65.0)
        if np.sum(kp_mask) >= 3:
            t_kp = t_v[kp_mask]
            y_kp = y_v[kp_mask]
        else:
            t_kp, y_kp = t_v, y_v

        popt, _ = curve_fit(korsmeyer_peppas, t_kp, y_kp, p0=[15.0, 0.45], bounds=((0, 0), (100, 2.0)))
        y_pred = korsmeyer_peppas(t_kp, *popt)
        r2_kp = max(0.0, float(r2_score(y_kp, y_pred)))
        n_val = float(popt[1])
        mech_info = classify_release_mechanism(n_val)

        results["Korsmeyer-Peppas"] = {
            "k_kp": round(float(popt[0]), 4),
            "release_exponent_n": round(n_val, 3),
            "R2": round(r2_kp, 4),
            "formula": "Q(t) = k * t^n",
            **mech_info
        }
    except Exception:
        results["Korsmeyer-Peppas"] = {
            "k_kp": 10.0, "release_exponent_n": 0.45, "R2": 0.0,
            **classify_release_mechanism(0.45)
        }

    # 5. Hixson-Crowell Fit
    try:
        popt, _ = curve_fit(hixson_crowell, t_v, y_v, p0=[0.01], bounds=(0, 1))
        y_pred = hixson_crowell(t_v, *popt)
        r2_hc = max(0.0, float(r2_score(y_v, y_pred)))
        results["Hixson-Crowell"] = {
            "rate_constant_khc": round(float(popt[0]), 4),
            "unit": "hour^-1",
            "R2": round(r2_hc, 4),
            "formula": "100^(1/3) - (100 - Q)^(1/3) = khc * t"
        }
    except Exception:
        results["Hixson-Crowell"] = {"rate_constant_khc": 0.0, "R2": 0.0, "unit": "hour^-1"}

    # Determine best fitting model based on R2
    best_model = max(["Zero-Order", "First-Order", "Higuchi", "Korsmeyer-Peppas", "Hixson-Crowell"],
                     key=lambda m: results[m].get("R2", 0.0))
    results["best_fitting_model"] = best_model
    results["best_R2"] = results[best_model].get("R2", 0.0)

    return results
