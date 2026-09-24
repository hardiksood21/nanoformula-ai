"""
Multi-Objective Formulation Optimizer and Pareto Frontier Engine.
Implements Non-Dominated Sorting, Desirability Functions, Applicability Domain Filtering,
and Diverse Recommendation Clustering.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


def calculate_desirability_size(pred_size: float, target_size: float, tolerance: float = 30.0) -> float:
    """
    Derringer-Suich desirability for particle size targeting:
    1.0 at exact target or smaller, decaying towards 0 as size exceeds target.
    """
    if pred_size <= target_size:
        # High desirability for smaller sizes, slight bonus for hitting target window
        return float(np.clip(1.0 - 0.2 * ((target_size - pred_size) / target_size), 0.7, 1.0))
    elif pred_size <= target_size + tolerance:
        return float(1.0 - ((pred_size - target_size) / tolerance))
    else:
        return float(max(0.01, 1.0 - ((pred_size - target_size) / (tolerance * 3.0))))


def calculate_desirability_ee(pred_ee: float, min_ee: float = 60.0) -> float:
    """
    Desirability function for Entrapment Efficiency:
    Higher EE% yields higher desirability (scale 0 to 1).
    """
    if pred_ee >= min_ee:
        return float(0.7 + 0.3 * ((pred_ee - min_ee) / (100.0 - min_ee)))
    else:
        return float(np.clip(pred_ee / min_ee * 0.7, 0.05, 0.7))


def find_pareto_front(df_candidates: pd.DataFrame, size_col: str = 'pred_size', ee_col: str = 'pred_EE') -> pd.DataFrame:
    """
    Extracts the non-dominated Pareto front (minimizing size, maximizing EE).
    """
    df = df_candidates.copy()
    # Sort by size ascending (better), then EE descending
    df = df.sort_values(by=[size_col, ee_col], ascending=[True, False]).reset_index(drop=True)
    
    pareto_indices = []
    max_ee_so_far = -1.0
    
    for idx, row in df.iterrows():
        ee = row[ee_col]
        if ee > max_ee_so_far:
            pareto_indices.append(idx)
            max_ee_so_far = ee
            
    df['is_pareto'] = False
    df.loc[pareto_indices, 'is_pareto'] = True
    return df


class PLGAFormulationOptimizer:
    """
    Generates, evaluates, and ranks PLGA nanoparticle formulation candidates.
    """
    def __init__(self, models_bundle: Dict[str, Any]):
        self.size_model = models_bundle['plga_size']
        self.ee_model = models_bundle['plga_ee']
        self.lc_model = models_bundle.get('plga_lc')
        self.ad_model = models_bundle['plga_ad']
        self.features = models_bundle['plga_features']
        self.dataset = models_bundle['plga_data']

    def optimize(
        self,
        drug_properties: Dict[str, float],
        target_size: float = 180.0,
        min_ee: float = 70.0,
        n_recommendations: int = 5,
        constrained_polymer_mw: Optional[float] = None,
        constrained_la_ga: Optional[float] = None,
        n_candidates: int = 20000,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Runs Monte Carlo space generation, ML inference with UQ & AD, and returns Pareto-ranked formulations.
        """
        np.random.seed(random_seed)
        
        # Define physical formulation space boundaries based on empirical data distribution
        df_base = self.dataset
        mw_pool = [1.2, 4.5, 7.0, 10.0, 14.0, 24.0, 30.0, 45.0, 50.0, 65.0, 80.0, 110.0, 200.0]
        if constrained_polymer_mw is not None:
            mw_pool = [float(constrained_polymer_mw)]
            
        laga_pool = [1.0, 1.5, 2.0, 2.33, 3.0, 4.0, 5.67] # 50:50, 65:35, 75:25, 85:15
        if constrained_la_ga is not None:
            laga_pool = [float(constrained_la_ga)]

        # Generate candidate formulations
        candidates = []
        for _ in range(n_candidates):
            p_mw = np.random.choice(mw_pool)
            laga = np.random.choice(laga_pool)
            drug_poly = float(np.random.choice(np.geomspace(0.01, 0.5, 50)))
            surf_conc = float(np.random.uniform(0.1, 2.5))
            surf_hlb = float(np.random.choice([12.0, 14.0, 16.0, 18.0, 20.5, 29.0]))
            aq_org = float(np.random.choice([2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]))
            ph = float(np.random.choice([-1.0, 0.0, 0.5, 1.0]))
            solv_pol = float(np.random.choice([4.0, 4.4, 5.1, 5.8, 6.2, 7.2]))

            cand = {
                **drug_properties,
                'polymer_MW': p_mw,
                'LA/GA': laga,
                'drug/polymer': drug_poly,
                'surfactant_concentration': surf_conc,
                'surfactant_HLB': surf_hlb,
                'aqueous/organic': aq_org,
                'pH': ph,
                'solvent_polarity_index': solv_pol
            }
            candidates.append(cand)

        df_cand = pd.DataFrame(candidates)
        
        # Predict Size with UQ
        pred_sizes, std_sizes, ci_l_size, ci_u_size = self.size_model.predict(df_cand, return_std=True)
        pred_ees, std_ees, ci_l_ee, ci_u_ee = self.ee_model.predict(df_cand, return_std=True)
        
        df_cand['pred_size'] = pred_sizes
        df_cand['pred_size_std'] = std_sizes
        df_cand['pred_size_ci95_l'] = ci_l_size
        df_cand['pred_size_ci95_u'] = ci_u_size

        df_cand['pred_EE'] = pred_ees
        df_cand['pred_ee_std'] = std_ees
        df_cand['pred_ee_ci95_l'] = ci_l_ee
        df_cand['pred_ee_ci95_u'] = ci_u_ee

        # Predict Loading Capacity if model available
        if self.lc_model:
            df_cand['pred_LC'] = self.lc_model.predict(df_cand, return_std=False)
        else:
            # Theoretical LC approx = (EE% * DrugPolymer) / (1 + EE% * DrugPolymer) * 100
            df_cand['pred_LC'] = (df_cand['pred_EE'] / 100.0 * df_cand['drug/polymer']) / (1.0 + df_cand['pred_EE'] / 100.0 * df_cand['drug/polymer']) * 100.0

        # Check Applicability Domain
        ad_res = self.ad_model.check(df_cand)
        df_cand['ad_status'] = [s['status'] for s in ad_res['samples']]
        df_cand['ad_score'] = [s['reliability_score'] for s in ad_res['samples']]
        df_cand['leverage'] = [s['leverage'] for s in ad_res['samples']]

        # Calculate Multi-Criteria Desirability Score
        d_size = np.array([calculate_desirability_size(s, target_size) for s in df_cand['pred_size']])
        d_ee = np.array([calculate_desirability_ee(e, min_ee) for e in df_cand['pred_EE']])
        d_rel = df_cand['ad_score'].values / 100.0

        # Weighted geometric mean desirability
        df_cand['desirability'] = (d_size**0.45) * (d_ee**0.35) * (d_rel**0.20)
        df_cand = find_pareto_front(df_cand, size_col='pred_size', ee_col='pred_EE')

        # Filter out extreme outliers and extract top diverse recommendations
        top_pool = df_cand[df_cand['pred_size'] <= (target_size * 1.3)].nlargest(n_recommendations * 5, 'desirability')
        if len(top_pool) < n_recommendations:
            top_pool = df_cand.nlargest(n_recommendations * 5, 'desirability')

        diverse = []
        used_combos = set()

        for _, row in top_pool.iterrows():
            combo_key = (round(row['polymer_MW'], 1), round(row['drug/polymer'], 2), round(row['surfactant_concentration'], 1))
            if combo_key not in used_combos or len(diverse) < 2:
                diverse.append(row)
                used_combos.add(combo_key)
            if len(diverse) >= n_recommendations:
                break

        if len(diverse) < n_recommendations:
            diverse = [r for _, r in top_pool.head(n_recommendations).iterrows()]

        df_recs = pd.DataFrame(diverse).sort_values('pred_size').reset_index(drop=True)

        return {
            "recommendations": df_recs,
            "all_candidates": df_cand,
            "pareto_candidates": df_cand[df_cand['is_pareto']],
            "n_candidates": len(df_cand),
            "target_size": target_size,
            "min_ee": min_ee
        }


class ChitosanFormulationOptimizer:
    """
    Optimizes Chitosan-TPP Nanoparticle formulations targeting Particle Size, PDI, and Zeta Potential.
    """
    def __init__(self, models_bundle: Dict[str, Any]):
        self.size_model = models_bundle['cs_size']
        self.pdi_model = models_bundle['cs_pdi']
        self.zeta_model = models_bundle['cs_zeta']
        self.ad_model = models_bundle['cs_ad']
        self.features = models_bundle['cs_features']
        self.dataset = models_bundle['cs_data']

    def optimize(
        self,
        target_size: float = 150.0,
        max_pdi: float = 0.25,
        min_zeta: float = 20.0,
        n_recommendations: int = 5,
        constrained_mw: Optional[float] = None,
        n_candidates: int = 10000,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        np.random.seed(random_seed)
        
        mw_options = [5.0, 20.0, 50.0, 310.0]
        if constrained_mw is not None:
            mw_options = [float(constrained_mw)]

        conc_options = np.linspace(0.15, 0.90, 30).tolist()
        tpp_options = np.linspace(0.15, 0.85, 30).tolist()

        candidates = []
        for _ in range(n_candidates):
            mw = float(np.random.choice(mw_options))
            conc = float(np.random.choice(conc_options))
            tpp = float(np.random.choice(tpp_options))

            cand = {
                'chitosan_MW': mw,
                'chitosan_conc': conc,
                'TPP_conc': tpp,
                'chitosan_TPP_ratio': conc / tpp,
                'conc_x_TPP': conc * tpp,
                'MW_x_conc': mw * conc,
                'MW_x_TPP': mw * tpp,
                'log_MW': np.log10(mw),
                'total_solute': conc + tpp,
                'chitosan_fraction': conc / (conc + tpp)
            }
            candidates.append(cand)

        df_cand = pd.DataFrame(candidates)
        
        pred_sizes, std_sizes, ci_l_size, ci_u_size = self.size_model.predict(df_cand, return_std=True)
        pred_pdis = self.pdi_model.predict(df_cand)
        pred_zetas = self.zeta_model.predict(df_cand)

        df_cand['pred_size'] = pred_sizes
        df_cand['pred_size_std'] = std_sizes
        df_cand['pred_size_ci95_l'] = ci_l_size
        df_cand['pred_size_ci95_u'] = ci_u_size
        df_cand['pred_PDI'] = pred_pdis
        df_cand['pred_zeta'] = pred_zetas

        ad_res = self.ad_model.check(df_cand)
        df_cand['ad_status'] = [s['status'] for s in ad_res['samples']]
        df_cand['ad_score'] = [s['reliability_score'] for s in ad_res['samples']]

        # Desirability scoring: Target Size + Low PDI + High Zeta Potential
        d_size = np.array([calculate_desirability_size(s, target_size, tolerance=25.0) for s in df_cand['pred_size']])
        d_pdi = np.clip(1.0 - (df_cand['pred_PDI'].values - 0.15) / 0.25, 0.1, 1.0)
        d_zeta = np.clip((df_cand['pred_zeta'].values - 10.0) / 20.0, 0.1, 1.0)
        d_rel = df_cand['ad_score'].values / 100.0

        df_cand['desirability'] = (d_size**0.40) * (d_pdi**0.25) * (d_zeta**0.20) * (d_rel**0.15)
        
        # Filter top diverse
        top_pool = df_cand.nlargest(n_recommendations * 5, 'desirability')
        diverse = []
        used_mws = set()

        for _, row in top_pool.iterrows():
            mw = row['chitosan_MW']
            if mw not in used_mws or len(diverse) < 2:
                diverse.append(row)
                used_mws.add(mw)
            if len(diverse) >= n_recommendations:
                break

        if len(diverse) < n_recommendations:
            diverse = [r for _, r in top_pool.head(n_recommendations).iterrows()]

        df_recs = pd.DataFrame(diverse).sort_values('pred_size').reset_index(drop=True)

        return {
            "recommendations": df_recs,
            "all_candidates": df_cand,
            "n_candidates": len(df_cand),
            "target_size": target_size
        }
