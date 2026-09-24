"""
Explainable AI (XAI) module using SHAP (SHapley Additive exPlanations)
for Nanoparticle Formulation Models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import shap
import matplotlib.pyplot as plt

FEATURE_DISPLAY_NAMES = {
    'polymer_MW': 'PLGA MW (kDa)',
    'LA/GA': 'Lactide/Glycolide Ratio',
    'mol_MW': 'Drug Mol. Weight (g/mol)',
    'mol_logP': 'Drug Lipophilicity (LogP)',
    'mol_TPSA': 'Drug Polar Surface Area (Å²)',
    'mol_melting_point': 'Drug Melting Point (°C)',
    'mol_Hacceptors': 'H-Bond Acceptors',
    'mol_Hdonors': 'H-Bond Donors',
    'mol_heteroatoms': 'Heteroatoms Count',
    'drug/polymer': 'Drug-to-Polymer Ratio',
    'surfactant_concentration': 'Surfactant Conc. (% w/v)',
    'surfactant_HLB': 'Surfactant HLB',
    'aqueous/organic': 'Aq/Org Phase Ratio',
    'pH': 'Aqueous Phase pH',
    'solvent_polarity_index': 'Solvent Polarity Index',
    # Chitosan features
    'chitosan_MW': 'Chitosan MW (kDa)',
    'chitosan_conc': 'Chitosan Conc. (mg/mL)',
    'TPP_conc': 'TPP Conc. (mg/mL)',
    'chitosan_TPP_ratio': 'CS:TPP Mass Ratio',
    'conc_x_TPP': 'CS × TPP Interaction',
    'MW_x_conc': 'MW × CS Conc.',
    'MW_x_TPP': 'MW × TPP Conc.',
    'log_MW': 'Log10(MW)',
    'total_solute': 'Total Solute (mg/mL)',
    'chitosan_fraction': 'Chitosan Mass Fraction'
}


class ModelExplainer:
    """
    Computes global and local SHAP explanations for formulation models.
    """
    def __init__(self, model, feature_names: List[str], target_name: str = "Target"):
        self.model = model
        self.feature_names = feature_names
        self.target_name = target_name
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        # Extract underlying tree model if ensemble
        base_model = self.model
        if hasattr(self.model, 'xgb_model') and self.model.xgb_model is not None:
            base_model = self.model.xgb_model
        elif hasattr(self.model, 'rf_model') and self.model.rf_model is not None:
            base_model = self.model.rf_model

        try:
            self.explainer = shap.TreeExplainer(base_model)
        except Exception:
            self.explainer = None

    def explain_instance(self, sample_dict: Dict[str, Any], top_n: int = 6) -> List[Dict[str, Any]]:
        """
        Explains an individual formulation prediction by returning the top positive and negative feature impacts.
        """
        if self.explainer is None:
            return []

        row = np.array([[sample_dict.get(f, 0.0) for f in self.feature_names]], dtype=float)
        shap_vals = self.explainer.shap_values(row)[0]
        base_val = float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.0

        explanations = []
        for feat, val, s_val in zip(self.feature_names, row[0], shap_vals):
            explanations.append({
                "feature": feat,
                "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "actual_value": round(float(val), 3),
                "shap_impact": round(float(s_val), 2),
                "abs_impact": abs(float(s_val)),
                "direction": "INCREASES" if s_val > 0 else "DECREASES"
            })

        explanations.sort(key=lambda x: x["abs_impact"], reverse=True)
        return explanations[:top_n]

    def plot_instance_breakdown(self, sample_dict: Dict[str, Any], title: Optional[str] = None):
        """
        Generates a matplotlib bar chart of SHAP feature contributions for the sample.
        """
        explanations = self.explain_instance(sample_dict, top_n=8)
        if not explanations:
            return None

        features = [e['display_name'] for e in reversed(explanations)]
        impacts = [e['shap_impact'] for e in reversed(explanations)]
        colors = ['#27AE60' if v < 0 else '#E74C3C' for v in impacts]
        
        # Invert color scheme for EE (increase is green, decrease is red)
        if "ee" in self.target_name.lower() or "lc" in self.target_name.lower():
            colors = ['#27AE60' if v > 0 else '#E74C3C' for v in impacts]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.barh(features, impacts, color=colors, edgecolor='black', alpha=0.85)
        ax.axvline(0, color='black', linewidth=1.2, linestyle='--')
        ax.set_xlabel(f"SHAP Impact on {self.target_name}", fontweight='bold')
        ax.set_title(title or f"Feature Attribution for {self.target_name}", fontweight='bold', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        return fig
