"""
4D Drug Release Kinetics Predictor.
Simulates time-dependent cumulative drug release curves based on biophysical
mechanisms (surface desorption, matrix diffusion, polymer ester hydrolysis)
and fits Korsmeyer-Peppas, Higuchi, and First-Order models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from .kinetic_models import fit_all_kinetic_models


STANDARD_TIME_POINTS = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0, 48.0, 72.0, 96.0, 120.0, 168.0] # Hours (up to 7 days)


class DrugReleasePredictor:
    """
    Predicts multi-phase drug release kinetics from polymeric nanoparticles
    accounting for burst release, matrix diffusion, and hydrolytic polymer degradation.
    """
    def __init__(self):
        pass

    def predict_release_curve(
        self,
        formulation: Dict[str, Any],
        release_ph: float = 7.4,
        time_points: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Simulates in vitro cumulative drug release (%) over specified time points (hours).
        """
        if time_points is None:
            time_points = STANDARD_TIME_POINTS

        t_arr = np.array(time_points, dtype=float)

        # Extract physicochemical and formulation parameters
        p_mw = float(formulation.get('polymer_MW', 30.0))
        laga = float(formulation.get('LA/GA', 1.0))
        drug_logp = float(formulation.get('mol_logP', 2.5))
        drug_mw = float(formulation.get('mol_MW', 350.0))
        particle_size = float(formulation.get('pred_size', 160.0))
        ee_percent = float(formulation.get('pred_EE', 75.0))
        surf_conc = float(formulation.get('surfactant_concentration', 1.0))

        # -------------------------------------------------------------
        # Phase 1: Surface Burst Release (0 - 4 hours)
        # -------------------------------------------------------------
        # High surface-to-volume ratio (smaller size), low EE%, and low LogP increase burst release
        size_factor = np.clip(200.0 / max(particle_size, 30.0), 0.7, 2.0)
        unentrapped_fraction = max(0.05, (100.0 - ee_percent) / 100.0)
        
        # Hydrophilic drugs partition near surface; hydrophobic drugs stay in core
        logp_barrier = np.clip(1.0 / (1.0 + np.exp(drug_logp - 1.5)), 0.15, 0.85)
        
        burst_max = np.clip((12.0 * size_factor * logp_barrier) + (unentrapped_fraction * 15.0), 5.0, 45.0)
        k_burst = 0.8  # Rapid desorption rate constant (h^-1)
        q_burst = burst_max * (1.0 - np.exp(-k_burst * t_arr))

        # -------------------------------------------------------------
        # Phase 2: Matrix Diffusion Phase (4 - 48 hours)
        # -------------------------------------------------------------
        # Governed by Stokes-Einstein diffusion: D inversely proportional to drug molecular radius ~ MW^(1/3)
        diff_base = 35.0 / (1.0 + (drug_mw / 350.0)**0.33)
        # Polymer chain entanglement density increases with polymer MW
        entanglement_factor = np.clip(30.0 / max(p_mw, 5.0), 0.4, 1.8)
        k_diff = (0.025 * entanglement_factor) / (1.0 + max(0.0, drug_logp * 0.2))
        
        q_diff = (100.0 - burst_max) * 0.55 * (1.0 - np.exp(-k_diff * t_arr))

        # -------------------------------------------------------------
        # Phase 3: Hydrolytic Polymer Matrix Degradation & Erosion (24 - 168+ hours)
        # -------------------------------------------------------------
        # PLGA 50:50 (LA/GA = 1.0) hydrolyzes fastest; higher LA content hydrolyzes slower
        degradation_rate = 0.008 / (1.0 + (laga - 1.0) * 0.6)
        
        # Acidic pH (endosomal / tumor microenvironment pH 5.0) catalyzes ester hydrolysis
        if release_ph < 6.5:
            degradation_rate *= (1.0 + (6.5 - release_ph) * 0.4)
            
        # Sigmoidal onset for bulk erosion
        t_lag = 18.0 * (p_mw / 25.0)**0.4 * (laga)**0.5
        erosion_fraction = 1.0 / (1.0 + np.exp(-degradation_rate * (t_arr - t_lag)))
        q_erosion = (100.0 - burst_max - 35.0) * erosion_fraction

        # Cumulative Release combination
        raw_cumulative = q_burst + q_diff + q_erosion
        # Enforce physical monotonicity and upper asymptotic bound
        cumulative = np.maximum.accumulate(np.clip(raw_cumulative, 0.0, 99.5))

        # Key kinetic milestone extraction
        burst_2h = float(cumulative[np.argmin(np.abs(t_arr - 2.0))])
        release_24h = float(cumulative[np.argmin(np.abs(t_arr - 24.0))])
        release_48h = float(cumulative[np.argmin(np.abs(t_arr - 48.0))])
        release_72h = float(cumulative[np.argmin(np.abs(t_arr - 72.0))])
        release_120h = float(cumulative[np.argmin(np.abs(t_arr - 120.0))])

        # Fit all mathematical kinetic models
        kinetic_fits = fit_all_kinetic_models(t_arr, cumulative)

        # Burst release safety assessment
        if burst_2h < 20.0:
            burst_risk = "Low Burst (Optimal Sustained Delivery)"
            risk_level = "LOW"
        elif 20.0 <= burst_2h < 35.0:
            burst_risk = "Moderate Burst (Acceptable for Most APIs)"
            risk_level = "MODERATE"
        else:
            burst_risk = "High Burst (Caution: Rapid Initial Dose Dumping Risk)"
            risk_level = "HIGH"

        # Construct tabular dataframe for export/plotting
        df_profile = pd.DataFrame({
            "Time (hours)": t_arr,
            "Cumulative Release (%)": np.round(cumulative, 2)
        })

        return {
            "time_points": t_arr.tolist(),
            "cumulative_release": np.round(cumulative, 2).tolist(),
            "profile_df": df_profile,
            "burst_2h_percent": round(burst_2h, 1),
            "release_24h_percent": round(release_24h, 1),
            "release_48h_percent": round(release_48h, 1),
            "release_72h_percent": round(release_72h, 1),
            "release_120h_percent": round(release_120h, 1),
            "burst_risk": burst_risk,
            "burst_risk_level": risk_level,
            "release_ph": release_ph,
            "kinetic_fits": kinetic_fits
        }
