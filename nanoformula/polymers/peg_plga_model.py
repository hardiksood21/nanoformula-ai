"""
PEG-PLGA Stealth Nanoparticle Modeling Engine.
Calculates surface PEGylation density, brush vs. mushroom regimes,
phagocytic/RES macrophage evasion, and prolonged circulation half-life.
"""

import numpy as np
from typing import Dict, Any


class PEGPLGAModel:
    """
    Models block copolymer PEG-PLGA nanoparticles for prolonged blood circulation.
    """
    def __init__(self):
        pass

    def evaluate_stealth_properties(
        self,
        plga_mw_kda: float = 40.0,
        peg_mw_kda: float = 5.0,
        peg_weight_fraction_percent: float = 10.0,
        core_diameter_nm: float = 120.0
    ) -> Dict[str, Any]:
        """
        Calculates surface grafting density (chains/nm²), Flory radius (Rf),
        conformation regime, and blood circulation half-life multiplier.
        """
        # Flory radius of PEG chain in water: Rf = a * N^(3/5) where a ~ 0.35 nm, N = MW_peg / 44
        n_monomers = (peg_mw_kda * 1000.0) / 44.05
        rf_nm = 0.35 * (n_monomers ** 0.6)

        # Total surface area of single spherical core
        radius_nm = core_diameter_nm / 2.0
        surface_area_nm2 = 4.0 * np.pi * (radius_nm ** 2)
        volume_nm3 = (4.0 / 3.0) * np.pi * (radius_nm ** 3)

        # Density of PLGA ~ 1.25 g/cm³ = 1.25 * 10^-21 g/nm³
        plga_density = 1.25e-21
        core_mass_g = volume_nm3 * plga_density

        # Mass of PEG on surface
        f_peg = peg_weight_fraction_percent / 100.0
        peg_mass_g = core_mass_g * (f_peg / (1.0 - f_peg))
        
        # Number of PEG chains on surface
        n_chains = (peg_mass_g / (peg_mw_kda * 1000.0)) * 6.022e23
        grafting_density = n_chains / surface_area_nm2  # chains / nm²

        # Distance between adjacent PEG graft sites: D = 2 * sqrt(1 / (pi * sigma))
        distance_between_chains_nm = 2.0 * np.sqrt(1.0 / (np.pi * max(grafting_density, 0.001)))

        # Transition regime: D < 2*Rf -> Dense Brush; D >= 2*Rf -> Mushroom
        overlap_threshold = 2.0 * rf_nm
        if distance_between_chains_nm < overlap_threshold:
            regime = "Dense Brush Regime (High Protein Shielding)"
            stealth_factor = "HIGH"
            opsonin_reduction_percent = round(np.clip(80.0 + (grafting_density * 40.0), 80.0, 96.0), 1)
            half_life_multiplier = round(np.clip(3.5 + (grafting_density * 8.0), 3.0, 12.0), 1)
        else:
            regime = "Mushroom Regime (Moderate Shielding)"
            stealth_factor = "MODERATE"
            opsonin_reduction_percent = round(np.clip(40.0 + (grafting_density * 50.0), 40.0, 79.0), 1)
            half_life_multiplier = round(np.clip(1.5 + (grafting_density * 5.0), 1.2, 3.4), 1)

        hydrodynamic_diameter_nm = round(core_diameter_nm + (2.0 * rf_nm), 1)

        return {
            "core_diameter_nm": core_diameter_nm,
            "peg_mw_kda": peg_mw_kda,
            "peg_weight_fraction": peg_weight_fraction_percent,
            "flory_radius_rf_nm": round(float(rf_nm), 2),
            "peg_grafting_density_chains_nm2": round(float(grafting_density), 3),
            "interchain_distance_nm": round(float(distance_between_chains_nm), 2),
            "conformation_regime": regime,
            "stealth_rating": stealth_factor,
            "opsonization_reduction_percent": opsonin_reduction_percent,
            "circulation_half_life_multiplier": f"{half_life_multiplier}x baseline PLGA",
            "effective_hydrodynamic_diameter_nm": hydrodynamic_diameter_nm
        }
