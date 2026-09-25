"""
mRNA / siRNA Lipid Nanoparticle (LNP) Formulation Optimizer and Recipe Engine.
Based on clinical nanomedicine standards (Comirnaty, Spikevax, Onpattro).
Optimizes 4-component lipid mixtures, N/P ratio, size, PDI, and microfluidic mixing protocols.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


LIPID_COMPONENTS_DATABASE = {
    "SM-102": {
        "name": "SM-102 (Heptadecan-9-yl 8-((2-hydroxyethyl)(6-oxo-6-(undecyloxy)hexyl)amino)octanoate)",
        "type": "Ionizable Cationic Lipid",
        "mol_weight": 710.18,
        "pKa": 6.68,
        "clinical_reference": "Moderna Spikevax COVID-19 Vaccine",
        "recommended_mol_percent": 50.0
    },
    "ALC-0315": {
        "name": "ALC-0315 (((4-hydroxybutyl)azanediyl)di(hexane-6,1-diyl) bis(2-hexyldecanoate))",
        "type": "Ionizable Cationic Lipid",
        "mol_weight": 766.29,
        "pKa": 6.09,
        "clinical_reference": "Pfizer/BioNTech Comirnaty Vaccine",
        "recommended_mol_percent": 46.3
    },
    "DLin-MC3-DMA": {
        "name": "DLin-MC3-DMA (MC3)",
        "type": "Ionizable Cationic Lipid",
        "mol_weight": 642.09,
        "pKa": 6.44,
        "clinical_reference": "Alnylam Onpattro (Patisiran) FDA Approved",
        "recommended_mol_percent": 50.0
    },
    "DSPC": {
        "name": "1,2-Distearoyl-sn-glycero-3-phosphocholine (DSPC)",
        "type": "Helper Phospholipid",
        "mol_weight": 790.15,
        "role": "Bilayer Structural Rigidity & Lamellar Stability",
        "recommended_mol_percent": 10.0
    },
    "DOPE": {
        "name": "1,2-Dioleoyl-sn-glycero-3-phosphoethanolamine (DOPE)",
        "type": "Helper Phospholipid (Fusogenic)",
        "mol_weight": 744.03,
        "role": "Promotes Hexagonal H-II Phase for Endosomal Membrane Disruption",
        "recommended_mol_percent": 10.0
    },
    "Cholesterol": {
        "name": "Cholesterol",
        "type": "Sterol Stabilizer",
        "mol_weight": 386.65,
        "role": "Fills Interstitial Membrane Cavities & Regulates Fluidity",
        "recommended_mol_percent": 38.5
    },
    "DMG-PEG2000": {
        "name": "1,2-Dimyristoyl-rac-glycero-3-methoxypolyethylene glycol-2000",
        "type": "PEGylated Lipid",
        "mol_weight": 2509.2,
        "role": "Steric Colloid Stabilization & Particle Size Control",
        "recommended_mol_percent": 1.5
    },
    "ALC-0159": {
        "name": "2-[(polyethylene glycol)-2000]-N,N-ditetradecylacetamide",
        "type": "PEGylated Lipid",
        "mol_weight": 2500.0,
        "role": "Steric Stabilization (Pfizer formulation)",
        "recommended_mol_percent": 1.6
    }
}


class LNPOptimizer:
    """
    Optimizes multi-component Lipid Nanoparticles for mRNA, siRNA, or nucleic acid delivery.
    """
    def __init__(self):
        pass

    def optimize_lnp(
        self,
        ionizable_lipid_name: str = "SM-102",
        helper_lipid_name: str = "DSPC",
        peg_lipid_name: str = "DMG-PEG2000",
        target_np_ratio: float = 6.0,
        target_size_nm: float = 80.0,
        mrna_dose_ug: float = 50.0,
        batch_volume_ml: float = 5.0
    ) -> Dict[str, Any]:
        """
        Calculates optimal lipid molar ratios, mass quantities, N/P ratio stoichiometry,
        predicted size, PDI, entrapment efficiency, and microfluidic mixing recipe.
        """
        ionizable = LIPID_COMPONENTS_DATABASE.get(ionizable_lipid_name, LIPID_COMPONENTS_DATABASE["SM-102"])
        helper = LIPID_COMPONENTS_DATABASE.get(helper_lipid_name, LIPID_COMPONENTS_DATABASE["DSPC"])
        chol = LIPID_COMPONENTS_DATABASE["Cholesterol"]
        peg = LIPID_COMPONENTS_DATABASE.get(peg_lipid_name, LIPID_COMPONENTS_DATABASE["DMG-PEG2000"])

        # Optimize PEG mol% to target size: Higher PEG% produces smaller particles
        # Empirical relationship: Size (nm) ~ 55 + (35 / PEG_mol%)
        peg_mol_pct = np.clip(35.0 / max(target_size_nm - 55.0, 10.0), 0.8, 3.0)
        
        # Molar composition (mol%)
        ionizable_mol_pct = 50.0
        helper_mol_pct = 10.0
        peg_mol_pct = round(float(peg_mol_pct), 2)
        chol_mol_pct = round(100.0 - ionizable_mol_pct - helper_mol_pct - peg_mol_pct, 2)

        # Average molecular weight of total lipid mix
        avg_lipid_mw = (
            (ionizable_mol_pct / 100.0) * ionizable["mol_weight"] +
            (helper_mol_pct / 100.0) * helper["mol_weight"] +
            (chol_mol_pct / 100.0) * chol["mol_weight"] +
            (peg_mol_pct / 100.0) * peg["mol_weight"]
        )

        # N/P Ratio Stoichiometry calculation:
        # 1 nmol mRNA nucleotide contains 1 phosphate group (~330 g/mol nucleotide)
        nmol_phosphate = (mrna_dose_ug / 330.0) * 1000.0  # nmol
        nmol_ionizable_required = nmol_phosphate * target_np_ratio  # nmol of ionizable lipid
        
        total_nmol_lipids = nmol_ionizable_required / (ionizable_mol_pct / 100.0)
        total_lipid_mass_mg = (total_nmol_lipids * avg_lipid_mw) / 1e6

        # Individual lipid masses (mg)
        m_ionizable_mg = round((total_nmol_lipids * (ionizable_mol_pct / 100.0) * ionizable["mol_weight"]) / 1e6, 3)
        m_helper_mg = round((total_nmol_lipids * (helper_mol_pct / 100.0) * helper["mol_weight"]) / 1e6, 3)
        m_chol_mg = round((total_nmol_lipids * (chol_mol_pct / 100.0) * chol["mol_weight"]) / 1e6, 3)
        m_peg_mg = round((total_nmol_lipids * (peg_mol_pct / 100.0) * peg["mol_weight"]) / 1e6, 3)

        # Predicted CQAs
        pred_size = round(55.0 + (35.0 / peg_mol_pct), 1)
        pred_pdi = round(0.08 + 0.02 * np.random.uniform(0.8, 1.2), 3)
        
        # EE% depends strongly on N/P ratio (steep sigmoidal binding above N/P = 3)
        pred_ee = round(float(np.clip(96.0 / (1.0 + np.exp(-1.5 * (target_np_ratio - 3.5))), 40.0, 98.5)), 1)
        
        # Zeta potential at physiological pH 7.4 vs endosomal pH 5.5
        zeta_ph74 = round(float(np.random.uniform(-4.5, 1.5)), 1)
        zeta_ph55 = round(float(np.random.uniform(12.0, 22.0)), 1)

        # Total Lipid-to-mRNA mass ratio
        lipid_to_mrna_ratio = round((total_lipid_mass_mg * 1000.0) / mrna_dose_ug, 1)

        # Microfluidic Mixing Parameters (Ethanol : Citrate Aqueous = 1 : 3)
        ethanol_vol_ml = round(batch_volume_ml * 0.25, 2)
        aqueous_vol_ml = round(batch_volume_ml * 0.75, 2)

        sop_steps = [
            f"1. Organic Lipid Phase Preparation: Accurately dissolve {m_ionizable_mg} mg {ionizable_lipid_name}, {m_helper_mg} mg {helper_lipid_name}, {m_chol_mg} mg Cholesterol, and {m_peg_mg} mg {peg_lipid_name} in {ethanol_vol_ml} mL of 100% anhydrous molecular biology grade ethanol.",
            f"2. Aqueous mRNA Phase Preparation: Dilute {mrna_dose_ug} µg of mRNA in {aqueous_vol_ml} mL of 50 mM Sodium Citrate buffer (pH 4.0 ± 0.05). Ensure RNAse-free conditions.",
            f"3. Microfluidic / T-Junction Impingement Mixing: Mix the Organic and Aqueous phases using a staggered herringbone or cross-flow microfluidic mixer at a 1:3 Flow Rate Ratio (FRR: 1 mL/min Ethanol to 3 mL/min Aqueous buffer; Total Flow Rate = 12-16 mL/min).",
            f"4. Ultrafiltration & Buffer Exchange: Immediately dilute the resulting LNP suspension with 10 volumes of 1X PBS (pH 7.4). Concentrate and buffer exchange via 100 kDa MWCO centrifugal ultrafiltration to remove all residual ethanol.",
            f"5. Characterization & Sterility: Measure hydrodynamic size and PDI using DLS. Quantify encapsulation efficiency (% EE) via RiboGreen / Quanti-iT RNA fluorescent dye binding assay with and without 1% Triton X-100."
        ]

        return {
            "ionizable_lipid": ionizable_lipid_name,
            "helper_lipid": helper_lipid_name,
            "cholesterol": "Cholesterol",
            "peg_lipid": peg_lipid_name,
            "molar_ratios": {
                ionizable_lipid_name: ionizable_mol_pct,
                helper_lipid_name: helper_mol_pct,
                "Cholesterol": chol_mol_pct,
                peg_lipid_name: peg_mol_pct
            },
            "lipid_masses_mg": {
                ionizable_lipid_name: m_ionizable_mg,
                helper_lipid_name: m_helper_mg,
                "Cholesterol": m_chol_mg,
                peg_lipid_name: m_peg_mg,
                "total_lipid_mass_mg": round(total_lipid_mass_mg, 3)
            },
            "mrna_dose_ug": mrna_dose_ug,
            "np_ratio": target_np_ratio,
            "lipid_to_mrna_mass_ratio": lipid_to_mrna_ratio,
            "predicted_cqas": {
                "pred_size_nm": pred_size,
                "pred_pdi": pred_pdi,
                "pred_ee_percent": pred_ee,
                "zeta_potential_ph74_mv": zeta_ph74,
                "zeta_potential_ph55_mv": zeta_ph55
            },
            "microfluidics_recipe": {
                "ethanol_phase_vol_ml": ethanol_vol_ml,
                "aqueous_phase_vol_ml": aqueous_vol_ml,
                "flow_rate_ratio_org_aq": "1 : 3",
                "citrate_buffer_ph": 4.0,
                "final_buffer": "1X PBS (pH 7.4)"
            },
            "sop_steps": sop_steps
        }
