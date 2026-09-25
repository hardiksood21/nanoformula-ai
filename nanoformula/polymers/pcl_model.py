"""
Polycaprolactone (PCL) Nanoparticle Model.
Models long-term sustained release (1 to 6 months) for semicrystalline,
slowly-degrading hydrophobic polyester delivery matrices.
"""

import numpy as np
from typing import Dict, Any


class PCLModel:
    """
    Models Polycaprolactone (PCL) nanoparticles for extended sustained delivery.
    """
    def __init__(self):
        pass

    def evaluate_pcl_formulation(
        self,
        pcl_mw_kda: float = 45.0,
        drug_logp: float = 3.5,
        drug_mw: float = 400.0,
        target_size_nm: float = 180.0
    ) -> Dict[str, Any]:
        """
        Predicts encapsulation capacity and multi-month sustained release profiles.
        """
        # PCL degradation half-life is 1-2 years due to high hydrophobicity & crystallinity
        degradation_half_life_months = round(12.0 * (pcl_mw_kda / 45.0)**0.5, 1)

        # High drug lipophilicity (LogP > 3) demonstrates exceptional compatibility with PCL core
        compatibility_score = round(float(np.clip(70.0 + (drug_logp * 5.0), 40.0, 98.0)), 1)
        pred_ee = round(float(np.clip(65.0 + (drug_logp * 6.0) - (drug_mw / 100.0), 50.0, 95.0)), 1)

        # 30-day cumulative release estimate
        release_30d = round(float(np.clip(25.0 / (1.0 + (drug_logp * 0.15)), 8.0, 40.0)), 1)
        release_90d = round(float(np.clip(release_30d * 2.2, 20.0, 75.0)), 1)
        release_180d = round(float(np.clip(release_90d * 1.4, 35.0, 92.0)), 1)

        return {
            "polymer": f"Polycaprolactone (PCL, MW {pcl_mw_kda:.0f} kDa)",
            "degradation_half_life": f"~{degradation_half_life_months} months",
            "drug_compatibility_score": f"{compatibility_score}%",
            "predicted_ee_percent": pred_ee,
            "release_milestones": {
                "30_days": f"{release_30d}%",
                "90_days": f"{release_90d}%",
                "180_days": f"{release_180d}%"
            },
            "best_use_case": "Long-term sustained implants, depot injections, and hydrophobic oncology therapeutics."
        }
