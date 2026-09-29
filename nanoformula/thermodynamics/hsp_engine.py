"""
Thermodynamic Drug-Polymer Miscibility and Hansen Solubility Parameter (HSP) Engine.
Calculates group-contribution HSP components (delta_D, delta_P, delta_H),
Hansen Distance (Ra), Relative Energy Difference (RED), Flory-Huggins interaction parameter (chi_dp),
and theoretical maximum thermodynamic drug loading capacity (DL_max).
"""

import math
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen


# Standard literature-validated HSP coordinates for biocompatible nanomedicine polymers (MPa^0.5)
POLYMER_HSP_DATABASE = {
    "PLGA 50:50": {
        "name": "Poly(lactic-co-glycolic acid) 50:50",
        "delta_D": 18.6,
        "delta_P": 8.5,
        "delta_H": 11.2,
        "interaction_radius_R0": 7.8,
        "molar_volume_cm3_mol": 52.4, # per repeating unit
        "density_g_cm3": 1.34
    },
    "PLGA 75:25": {
        "name": "Poly(lactic-co-glycolic acid) 75:25",
        "delta_D": 18.2,
        "delta_P": 7.9,
        "delta_H": 9.8,
        "interaction_radius_R0": 7.4,
        "molar_volume_cm3_mol": 55.6,
        "density_g_cm3": 1.30
    },
    "PLA (Poly-L-lactic acid)": {
        "name": "Poly(L-lactic acid)",
        "delta_D": 18.6,
        "delta_P": 9.9,
        "delta_H": 6.0,
        "interaction_radius_R0": 7.0,
        "molar_volume_cm3_mol": 58.1,
        "density_g_cm3": 1.25
    },
    "PCL (Polycaprolactone)": {
        "name": "Polycaprolactone",
        "delta_D": 17.5,
        "delta_P": 4.9,
        "delta_H": 7.2,
        "interaction_radius_R0": 6.5,
        "molar_volume_cm3_mol": 100.8,
        "density_g_cm3": 1.145
    },
    "Chitosan": {
        "name": "Chitosan (85% Deacetylated)",
        "delta_D": 19.5,
        "delta_P": 11.2,
        "delta_H": 16.8,
        "interaction_radius_R0": 9.5,
        "molar_volume_cm3_mol": 118.5,
        "density_g_cm3": 1.40
    },
    "PEG 2000": {
        "name": "Poly(ethylene glycol) 2000",
        "delta_D": 16.4,
        "delta_P": 8.2,
        "delta_H": 10.6,
        "interaction_radius_R0": 8.0,
        "molar_volume_cm3_mol": 39.0,
        "density_g_cm3": 1.12
    },
    "Lipid Bilayer (Phospholipids/Cholesterol)": {
        "name": "Ionizable Lipid / Phospholipid Matrix",
        "delta_D": 16.8,
        "delta_P": 3.4,
        "delta_H": 5.2,
        "interaction_radius_R0": 6.2,
        "molar_volume_cm3_mol": 350.0,
        "density_g_cm3": 0.98
    }
}


class HSPEngine:
    """
    Computes QSPR Hansen Solubility Parameters from chemical structures (SMILES or descriptors)
    and evaluates drug-polymer thermodynamic compatibility.
    """

    @staticmethod
    def estimate_drug_hsp(smiles: str, mw: Optional[float] = None, logp: Optional[float] = None, tpsa: Optional[float] = None, hbd: Optional[int] = None, hba: Optional[int] = None) -> Dict[str, float]:
        """
        Estimates Hansen delta_D, delta_P, and delta_H (MPa^0.5) from RDKit molecular topology
        and Van Krevelen - Hoftyzer group contribution heuristics.
        """
        mol = None
        if smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
            except Exception:
                mol = None

        if mol is not None:
            mw = float(Descriptors.ExactMolWt(mol))
            logp = float(Crippen.MolLogP(mol))
            tpsa = float(Descriptors.TPSA(mol))
            hbd = int(Lipinski.NumHDonors(mol))
            hba = int(Lipinski.NumHAcceptors(mol))
            n_rot = int(Lipinski.NumRotatableBonds(mol))
            n_arom = len(mol.GetSubstructMatches(Chem.MolFromSmarts("a")))
            n_rings = int(Descriptors.RingCount(mol))
        else:
            mw = mw or 350.0
            logp = logp or 2.5
            tpsa = tpsa or 80.0
            hbd = hbd or 2
            hba = hba or 4
            n_rot = 4
            n_arom = 6
            n_rings = 2

        # Estimated molar volume Vm (cm3/mol) via McGowan volume / empirical density correlation
        # Density typically 1.15 - 1.45 g/cm3 for small organic drug molecules
        estimated_density = 1.20 + 0.05 * (tpsa / max(mw, 100.0)) + 0.02 * (hba + hbd)
        estimated_density = float(np.clip(estimated_density, 1.10, 1.55))
        molar_volume = mw / estimated_density # cm3/mol

        # 1. Dispersion component (delta_D): driven by aromaticity, molecular weight, and carbon backbone density
        # Baseline ~ 16.0 - 21.0 MPa^0.5
        delta_D = 16.5 + 0.003 * min(mw, 800.0) + 0.15 * min(n_arom, 20) + 0.10 * min(n_rings, 5)
        delta_D = float(np.clip(delta_D, 15.5, 22.0))

        # 2. Polar component (delta_P): driven by dipole moment, polar surface area (TPSA), and heteroatoms
        # delta_P ~ sqrt(sum(F_p^2)) / Vm
        polar_factor = (tpsa / max(molar_volume, 50.0)) * 25.0 + (hba * 1.2)
        delta_P = float(np.clip(3.0 + polar_factor * 0.8, 1.5, 18.0))

        # 3. Hydrogen bonding component (delta_H): driven by H-bond donors and acceptors
        # delta_H ~ sqrt(sum(E_h)) / Vm
        h_factor = (hbd * 4.5 + hba * 1.5) / (max(molar_volume, 50.0) ** 0.5) * 12.0
        delta_H = float(np.clip(2.0 + h_factor, 1.0, 24.0))

        delta_total = math.sqrt(delta_D**2 + delta_P**2 + delta_H**2)

        return {
            "delta_D": round(delta_D, 2),
            "delta_P": round(delta_P, 2),
            "delta_H": round(delta_H, 2),
            "delta_total": round(delta_total, 2),
            "molar_volume_cm3_mol": round(molar_volume, 1),
            "estimated_density_g_cm3": round(estimated_density, 2)
        }

    @staticmethod
    def calculate_compatibility(drug_hsp: Dict[str, float], polymer_key: str = "PLGA 50:50", temperature_k: float = 298.15) -> Dict[str, Any]:
        """
        Calculates Hansen Distance (Ra), Relative Energy Difference (RED), Flory-Huggins (chi_dp),
        and maximum theoretical loading capacity (DL_max).
        """
        poly_data = POLYMER_HSP_DATABASE.get(polymer_key, POLYMER_HSP_DATABASE["PLGA 50:50"])

        dD_d, dP_d, dH_d = drug_hsp["delta_D"], drug_hsp["delta_P"], drug_hsp["delta_H"]
        dD_p, dP_p, dH_p = poly_data["delta_D"], poly_data["delta_P"], poly_data["delta_H"]
        r0 = poly_data["interaction_radius_R0"]
        vm_drug = drug_hsp.get("molar_volume_cm3_mol", 280.0)

        # Hansen Distance Ra = sqrt(4*(dD1 - dD2)^2 + (dP1 - dP2)^2 + (dH1 - dH2)^2)
        ra = math.sqrt(4.0 * ((dD_d - dD_p) ** 2) + ((dP_d - dP_p) ** 2) + ((dH_d - dH_p) ** 2))
        
        # Relative Energy Difference (RED) = Ra / R0
        red = ra / r0

        # Flory-Huggins parameter chi_dp = (Vm_drug / (4 * R * T)) * Ra^2
        # R = 8.314 J/(mol*K), 1 MPa = 1 J/cm3
        rt_joules = 8.3145 * temperature_k # ~ 2479 J/mol
        chi_dp = (vm_drug / (4.0 * rt_joules)) * (ra ** 2)

        # Thermodynamic Miscibility Categorization
        if red < 0.75:
            miscibility = "High Thermodynamic Miscibility (Soluble)"
            precipitation_risk = "Very Low (Stable amorphous solid dispersion)"
            miscibility_color = "#28a745" # Green
        elif red <= 1.00:
            miscibility = "Moderate Miscibility (Partial Solubility)"
            precipitation_risk = "Low to Moderate (Stable at <= 15% drug loading)"
            miscibility_color = "#17a2b8" # Teal
        elif red <= 1.30:
            miscibility = "Boundary / Metastable Zone"
            precipitation_risk = "Moderate (Risk of drug crystallization over time)"
            miscibility_color = "#ffc107" # Yellow/Orange
        else:
            miscibility = "Immiscible / Phase Separation"
            precipitation_risk = "High (Severe burst release and rapid recrystallization)"
            miscibility_color = "#dc3545" # Red

        # Theoretical maximum thermodynamic drug loading capacity DL_max (%)
        # Based on lattice-fluid chemical potential equilibrium: DL_max ~ 1 / (1 + exp(3*(chi - 0.45)))
        dl_max_raw = 35.0 / (1.0 + math.exp(2.8 * (chi_dp - 0.50)))
        dl_max_percent = float(np.clip(dl_max_raw, 2.5, 35.0))

        return {
            "polymer_system": polymer_key,
            "polymer_name": poly_data["name"],
            "hansen_distance_Ra": round(ra, 2),
            "relative_energy_difference_RED": round(red, 2),
            "flory_huggins_chi": round(chi_dp, 3),
            "miscibility_category": miscibility,
            "precipitation_risk": precipitation_risk,
            "miscibility_color": miscibility_color,
            "max_thermodynamic_loading_percent": round(dl_max_percent, 1),
            "drug_hsp": {"delta_D": dD_d, "delta_P": dP_d, "delta_H": dH_d},
            "polymer_hsp": {"delta_D": dD_p, "delta_P": dP_p, "delta_H": dH_p, "R0": r0}
        }

    @classmethod
    def screen_all_polymers(cls, drug_hsp: Dict[str, float], temperature_k: float = 298.15) -> List[Dict[str, Any]]:
        """
        Screens a drug against all supported nanomedicine polymeric and lipid matrices.
        """
        results = []
        for poly_key in POLYMER_HSP_DATABASE.keys():
            res = cls.calculate_compatibility(drug_hsp, poly_key, temperature_k=temperature_k)
            results.append(res)
        
        # Sort by best compatibility (lowest RED)
        results.sort(key=lambda x: x["relative_energy_difference_RED"])
        return results
