"""
Active Learning & Lab-in-the-Loop Bayesian Feedback Engine.
Ingests real experimental batches from wet-lab researchers, logs experimental provenance,
updates surrogate Gaussian Process models, and computes Expected Improvement (EI) recommendations.
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


FEEDBACK_DB_PATH = os.path.join("data", "experimental_feedback_db.json")


class ActiveLearningEngine:
    """
    Manages wet-lab experimental feedback ingestion and Bayesian surrogate model updating.
    """
    def __init__(self, db_path: str = FEEDBACK_DB_PATH):
        self.db_path = db_path
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w") as f:
                json.dump({"version": "1.0", "entries": []}, f, indent=2)

    def get_all_entries(self) -> List[Dict[str, Any]]:
        """Returns all recorded experimental feedback entries."""
        try:
            with open(self.db_path, "r") as f:
                data = json.load(f)
                return data.get("entries", [])
        except Exception:
            return []

    def submit_experimental_batch(
        self,
        researcher_name: str,
        institution: str,
        polymer_system: str,
        drug_name: str,
        formulation_parameters: Dict[str, float],
        measured_particle_size_nm: float,
        measured_ee_percent: Optional[float] = None,
        measured_pdi: Optional[float] = None,
        measured_zeta_potential_mv: Optional[float] = None,
        instrument_notes: str = "DLS / Malvern Zetasizer"
    ) -> Dict[str, Any]:
        """
        Records a real wet-lab batch result with timestamp and metadata into the local database.
        """
        entries = self.get_all_entries()
        
        entry = {
            "entry_id": f"BATCH_{len(entries) + 1:04d}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "researcher": researcher_name,
            "institution": institution,
            "polymer_system": polymer_system,
            "drug_name": drug_name,
            "formulation_parameters": formulation_parameters,
            "experimental_results": {
                "particle_size_nm": float(measured_particle_size_nm),
                "ee_percent": float(measured_ee_percent) if measured_ee_percent is not None else None,
                "pdi": float(measured_pdi) if measured_pdi is not None else None,
                "zeta_potential_mv": float(measured_zeta_potential_mv) if measured_zeta_potential_mv is not None else None
            },
            "instrument_notes": instrument_notes
        }

        entries.append(entry)
        with open(self.db_path, "w") as f:
            json.dump({"version": "1.0", "total_entries": len(entries), "entries": entries}, f, indent=2)

        return {
            "status": "SUCCESS",
            "message": f"Experimental batch {entry['entry_id']} successfully recorded in active learning repository.",
            "entry": entry
        }

    def compute_active_learning_metrics(self) -> Dict[str, Any]:
        """Computes summary statistics on wet-lab data contributions."""
        entries = self.get_all_entries()
        if not entries:
            return {
                "total_batches_contributed": 0,
                "unique_drugs_tested": 0,
                "status": "No user experimental batches registered yet."
            }

        drugs = set(e.get("drug_name", "") for e in entries)
        polymers = set(e.get("polymer_system", "") for e in entries)
        
        return {
            "total_batches_contributed": len(entries),
            "unique_drugs_tested": len(drugs),
            "polymer_systems": list(polymers),
            "status": f"{len(entries)} experimental batch(es) available for Bayesian surrogate updates."
        }

    def fit_bayesian_surrogate(self, X_base: np.ndarray, y_base: np.ndarray, feature_names: List[str]) -> GaussianProcessRegressor:
        """
        Fits a Gaussian Process surrogate with Matern 5/2 kernel combining base data
        and user experimental batches to compute Expected Improvement (EI).
        """
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=42)
        
        # Include feedback entries if compatible
        entries = self.get_all_entries()
        extra_X, extra_y = [], []
        for e in entries:
            params = e.get("formulation_parameters", {})
            if all(f in params for f in feature_names):
                val_y = e.get("experimental_results", {}).get("particle_size_nm")
                if val_y is not None:
                    extra_X.append([params[f] for f in feature_names])
                    extra_y.append(float(val_y))

        if extra_X:
            X_all = np.vstack([X_base, np.array(extra_X)])
            y_all = np.concatenate([y_base, np.array(extra_y)])
        else:
            X_all = X_base
            y_all = y_base

        gp.fit(X_all, y_all)
        return gp
