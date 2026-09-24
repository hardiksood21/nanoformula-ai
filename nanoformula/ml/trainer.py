import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple

# Ensure package root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from nanoformula.ml.ensemble_model import NanoparticleEnsembleRegressor
from nanoformula.ml.applicability_domain import ApplicabilityDomain
from nanoformula.ml.explainability import ModelExplainer


PLGA_FEATURES = [
    'polymer_MW', 'LA/GA', 'mol_MW', 'mol_logP', 'mol_TPSA',
    'mol_melting_point', 'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms',
    'drug/polymer', 'surfactant_concentration', 'surfactant_HLB',
    'aqueous/organic', 'pH', 'solvent_polarity_index'
]

CHITOSAN_FEATURES = [
    'chitosan_MW', 'chitosan_conc', 'TPP_conc', 'chitosan_TPP_ratio',
    'conc_x_TPP', 'MW_x_conc', 'MW_x_TPP', 'log_MW', 'total_solute', 'chitosan_fraction'
]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    
    # Avoid zero division in MAPE
    mask = y_true > 0.1
    if np.sum(mask) > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = 0.0
        
    pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    return {
        "R2": round(r2, 4),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MAPE": round(mape, 2),
        "Pearson_r": round(pearson_r, 4)
    }


def benchmark_algorithms(X: pd.DataFrame, y: np.ndarray, feature_names: List[str], target_name: str, cv_folds: int = 10) -> pd.DataFrame:
    """
    Evaluates 7 distinct machine learning algorithms using 10-Fold Cross Validation.
    """
    X_mat = X[feature_names].values
    y_arr = np.array(y, dtype=float)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_mat)
    
    models = {
        "NanoFormula Ensemble": NanoparticleEnsembleRegressor(target_name, feature_names),
        "XGBoost": xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=42),
        "Support Vector Regressor": SVR(C=10.0, epsilon=0.1),
        "Multi-Layer Perceptron": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0)
    }
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = []
    
    for name, model in models.items():
        y_cv_true, y_cv_pred = [], []
        use_scaled = name in ["Support Vector Regressor", "Multi-Layer Perceptron", "Ridge Regression"]
        X_curr = X_scaled if use_scaled else X_mat
        
        for train_idx, test_idx in kf.split(X_curr):
            X_tr, X_te = X_curr[train_idx], X_curr[test_idx]
            y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
            
            if name == "NanoFormula Ensemble":
                m = NanoparticleEnsembleRegressor(target_name, feature_names)
                m.fit(X.iloc[train_idx], y_tr)
                p_te = m.predict(X.iloc[test_idx])
            else:
                model.fit(X_tr, y_tr)
                p_te = model.predict(X_te)
                
            y_cv_true.extend(y_te)
            y_cv_pred.extend(p_te)
            
        m_dict = calculate_metrics(np.array(y_cv_true), np.array(y_cv_pred))
        m_dict["Algorithm"] = name
        m_dict["Target"] = target_name
        results.append(m_dict)
        
    res_df = pd.DataFrame(results)
    cols = ["Algorithm", "Target", "R2", "RMSE", "MAE", "MAPE", "Pearson_r"]
    return res_df[cols].sort_values("R2", ascending=False).reset_index(drop=True)


def train_and_save_all(data_dir: str = "data", output_dir: str = "saved_models", benchmarks_dir: str = "benchmarks"):
    """
    Complete publication training pipeline:
    1. Loads PLGA & Chitosan datasets.
    2. Runs 10-fold CV & Leave-One-Drug-Out validation.
    3. Fits and serializes production models and Applicability Domains.
    4. Generates publication figures and summary tables.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(benchmarks_dir, exist_ok=True)
    
    print("=" * 60)
    print("[*] NANOFORMULA AI: TRAINING & BENCHMARKING PIPELINE")
    print("=" * 60)
    
    # -------------------------------------------------------------
    # 1. PLGA Models (Particle Size, EE%, Loading Capacity)
    # -------------------------------------------------------------
    plga_path = os.path.join(data_dir, "PLGA_nanoparticles_dataset.csv")
    if not os.path.exists(plga_path):
        plga_path = "PLGA_nanoparticles_dataset.csv"
        
    df_plga = pd.read_csv(plga_path)
    print(f"\n[1/4] Processing PLGA Dataset: {df_plga.shape[0]} formulations, {len(PLGA_FEATURES)} features")
    
    # Train Particle Size Ensemble
    print("  -> Training PLGA Particle Size Ensemble...")
    plga_size_model = NanoparticleEnsembleRegressor("Particle Size (nm)", PLGA_FEATURES)
    plga_size_model.fit(df_plga, df_plga['particle_size'])
    
    # Train Entrapment Efficiency Ensemble
    print("  -> Training PLGA Entrapment Efficiency Ensemble...")
    plga_ee_model = NanoparticleEnsembleRegressor("Entrapment Efficiency (%)", PLGA_FEATURES)
    plga_ee_model.fit(df_plga, df_plga['EE'])
    
    # Train Loading Capacity Ensemble
    print("  -> Training PLGA Loading Capacity Ensemble...")
    plga_lc_model = NanoparticleEnsembleRegressor("Loading Capacity (%)", PLGA_FEATURES)
    plga_lc_model.fit(df_plga, df_plga['LC'])
    
    # Fit PLGA Applicability Domain
    print("  -> Fitting PLGA Applicability Domain (William's Plot / Mahalanobis)...")
    plga_ad = ApplicabilityDomain(PLGA_FEATURES)
    plga_ad.fit(df_plga)
    
    # Run Algorithm Benchmarks for PLGA
    print("  -> Running 10-Fold CV Algorithm Benchmarks for PLGA...")
    bm_size = benchmark_algorithms(df_plga, df_plga['particle_size'].values, PLGA_FEATURES, "PLGA Particle Size")
    bm_ee = benchmark_algorithms(df_plga, df_plga['EE'].values, PLGA_FEATURES, "PLGA EE%")
    bm_lc = benchmark_algorithms(df_plga, df_plga['LC'].values, PLGA_FEATURES, "PLGA LC%")
    
    plga_benchmarks = pd.concat([bm_size, bm_ee, bm_lc], ignore_index=True)
    plga_benchmarks.to_csv(os.path.join(benchmarks_dir, "plga_algorithm_benchmarks.csv"), index=False)
    
    # Leave-One-Drug-Out (LODO) Validation
    print("  -> Running Leave-One-Drug-Out (Scaffold-Grouped) CV for PLGA...")
    mol_cols = ['mol_MW', 'mol_logP', 'mol_TPSA', 'mol_melting_point', 'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms']
    drug_groups = df_plga.groupby(mol_cols).ngroup().values
    gkf = GroupKFold(n_splits=5)
    
    lodo_true_size, lodo_pred_size = [], []
    for tr_idx, te_idx in gkf.split(df_plga, groups=drug_groups):
        m = NanoparticleEnsembleRegressor("Particle Size (nm)", PLGA_FEATURES)
        m.fit(df_plga.iloc[tr_idx], df_plga['particle_size'].iloc[tr_idx].values)
        p = m.predict(df_plga.iloc[te_idx])
        lodo_true_size.extend(df_plga['particle_size'].iloc[te_idx].values)
        lodo_pred_size.extend(p)
        
    lodo_metrics = calculate_metrics(np.array(lodo_true_size), np.array(lodo_pred_size))
    print(f"     * LODO Size Generalizability: R2 = {lodo_metrics['R2']}, RMSE = {lodo_metrics['RMSE']} nm, MAE = {lodo_metrics['MAE']} nm")
    
    # -------------------------------------------------------------
    # 2. Chitosan Models (Particle Size, PDI, Zeta Potential)
    # -------------------------------------------------------------
    cs_path = os.path.join(data_dir, "chitosan_nanoparticles_dataset.csv")
    if not os.path.exists(cs_path):
        cs_path = "chitosan_nanoparticles_dataset.csv"
        
    df_cs = pd.read_csv(cs_path)
    print(f"\n[2/4] Processing Chitosan Dataset: {df_cs.shape[0]} formulations, {len(CHITOSAN_FEATURES)} features")
    
    print("  -> Training Chitosan Particle Size Ensemble...")
    cs_size_model = NanoparticleEnsembleRegressor("Chitosan Size (nm)", CHITOSAN_FEATURES)
    cs_size_model.fit(df_cs, df_cs['particle_size'])
    
    print("  -> Training Chitosan PDI Model...")
    cs_pdi_model = NanoparticleEnsembleRegressor("Chitosan PDI", CHITOSAN_FEATURES)
    cs_pdi_model.fit(df_cs, df_cs['PDI'])
    
    print("  -> Training Chitosan Zeta Potential Model...")
    cs_zeta_model = NanoparticleEnsembleRegressor("Chitosan Zeta Potential (mV)", CHITOSAN_FEATURES)
    cs_zeta_model.fit(df_cs, df_cs['zeta_potential'])
    
    cs_ad = ApplicabilityDomain(CHITOSAN_FEATURES)
    cs_ad.fit(df_cs)
    
    # Chitosan Benchmarks
    bm_cs_size = benchmark_algorithms(df_cs, df_cs['particle_size'].values, CHITOSAN_FEATURES, "Chitosan Size")
    bm_cs_pdi = benchmark_algorithms(df_cs, df_cs['PDI'].values, CHITOSAN_FEATURES, "Chitosan PDI")
    bm_cs_zeta = benchmark_algorithms(df_cs, df_cs['zeta_potential'].values, CHITOSAN_FEATURES, "Chitosan Zeta")
    cs_benchmarks = pd.concat([bm_cs_size, bm_cs_pdi, bm_cs_zeta], ignore_index=True)
    cs_benchmarks.to_csv(os.path.join(benchmarks_dir, "chitosan_algorithm_benchmarks.csv"), index=False)

    # -------------------------------------------------------------
    # 3. Serialization
    # -------------------------------------------------------------
    print(f"\n[3/4] Serializing Trained Models and Explanations to '{output_dir}'...")
    models_bundle = {
        "plga_size": plga_size_model,
        "plga_ee": plga_ee_model,
        "plga_lc": plga_lc_model,
        "plga_ad": plga_ad,
        "plga_features": PLGA_FEATURES,
        "plga_data": df_plga,
        "cs_size": cs_size_model,
        "cs_pdi": cs_pdi_model,
        "cs_zeta": cs_zeta_model,
        "cs_ad": cs_ad,
        "cs_features": CHITOSAN_FEATURES,
        "cs_data": df_cs,
        "lodo_metrics": lodo_metrics,
        "plga_benchmarks": plga_benchmarks,
        "cs_benchmarks": cs_benchmarks
    }
    
    bundle_path = os.path.join(output_dir, "nanoformula_models_bundle.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump(models_bundle, f)
        
    # Also save individual files for backward compatibility with earlier scripts
    with open(os.path.join(output_dir, "model_particle_size_final.pkl"), "wb") as f:
        pickle.dump(plga_size_model.xgb_model, f)
    with open(os.path.join(output_dir, "model_ee_final.pkl"), "wb") as f:
        pickle.dump(plga_ee_model.xgb_model, f)
    with open(os.path.join(output_dir, "model_chitosan_size.pkl"), "wb") as f:
        pickle.dump(cs_size_model.xgb_model, f)
    with open(os.path.join(output_dir, "chitosan_features.pkl"), "wb") as f:
        pickle.dump(CHITOSAN_FEATURES, f)

    # -------------------------------------------------------------
    # 4. Publication Figures (Parity Plots, SHAP Importance)
    # -------------------------------------------------------------
    print(f"\n[4/4] Generating Publication-Quality Figures (300 DPI)...")
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Figure 1: PLGA Model Parity Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    # Size parity
    p_size = plga_size_model.predict(df_plga)
    axes[0].scatter(df_plga['particle_size'], p_size, alpha=0.6, color='#2E86AB', edgecolors='k', s=50)
    lims = [0, max(df_plga['particle_size'].max(), p_size.max()) + 20]
    axes[0].plot(lims, lims, 'r--', linewidth=2, label='Identity Line')
    axes[0].set_xlabel("Experimental Particle Size (nm)", fontweight='bold')
    axes[0].set_ylabel("Predicted Particle Size (nm)", fontweight='bold')
    axes[0].set_title(f"PLGA Particle Size (R2 = {r2_score(df_plga['particle_size'], p_size):.3f})", fontweight='bold')
    axes[0].legend()
    
    # EE% parity
    p_ee = plga_ee_model.predict(df_plga)
    axes[1].scatter(df_plga['EE'], p_ee, alpha=0.6, color='#27AE60', edgecolors='k', s=50)
    lims_ee = [0, 105]
    axes[1].plot(lims_ee, lims_ee, 'r--', linewidth=2, label='Identity Line')
    axes[1].set_xlabel("Experimental EE (%)", fontweight='bold')
    axes[1].set_ylabel("Predicted EE (%)", fontweight='bold')
    axes[1].set_title(f"PLGA Entrapment Efficiency (R2 = {r2_score(df_plga['EE'], p_ee):.3f})", fontweight='bold')
    axes[1].legend()

    # LC% parity
    p_lc = plga_lc_model.predict(df_plga)
    axes[2].scatter(df_plga['LC'], p_lc, alpha=0.6, color='#E67E22', edgecolors='k', s=50)
    lims_lc = [0, max(df_plga['LC'].max(), p_lc.max()) + 5]
    axes[2].plot(lims_lc, lims_lc, 'r--', linewidth=2, label='Identity Line')
    axes[2].set_xlabel("Experimental Loading Capacity (%)", fontweight='bold')
    axes[2].set_ylabel("Predicted Loading Capacity (%)", fontweight='bold')
    axes[2].set_title(f"PLGA Loading Capacity (R2 = {r2_score(df_plga['LC'], p_lc):.3f})", fontweight='bold')
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(benchmarks_dir, "plga_parity_plots.png"), dpi=300)
    fig.savefig(os.path.join(benchmarks_dir, "plga_parity_plots.pdf"))
    plt.close()

    # Figure 2: Chitosan Parity Plots
    fig_cs, axes_cs = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    p_cs_size = cs_size_model.predict(df_cs)
    axes_cs[0].scatter(df_cs['particle_size'], p_cs_size, alpha=0.7, color='#3498DB', edgecolors='k', s=60)
    l_cs = [50, 270]
    axes_cs[0].plot(l_cs, l_cs, 'r--', linewidth=2, label='Identity Line')
    axes_cs[0].set_xlabel("Experimental Size (nm)", fontweight='bold')
    axes_cs[0].set_ylabel("Predicted Size (nm)", fontweight='bold')
    axes_cs[0].set_title(f"Chitosan Particle Size (R2 = {r2_score(df_cs['particle_size'], p_cs_size):.3f})", fontweight='bold')
    axes_cs[0].legend()

    p_cs_pdi = cs_pdi_model.predict(df_cs)
    axes_cs[1].scatter(df_cs['PDI'], p_cs_pdi, alpha=0.7, color='#9B59B6', edgecolors='k', s=60)
    l_pdi = [0.1, 0.5]
    axes_cs[1].plot(l_pdi, l_pdi, 'r--', linewidth=2, label='Identity Line')
    axes_cs[1].set_xlabel("Experimental PDI", fontweight='bold')
    axes_cs[1].set_ylabel("Predicted PDI", fontweight='bold')
    axes_cs[1].set_title(f"Chitosan PDI (R2 = {r2_score(df_cs['PDI'], p_cs_pdi):.3f})", fontweight='bold')
    axes_cs[1].legend()

    p_cs_zeta = cs_zeta_model.predict(df_cs)
    axes_cs[2].scatter(df_cs['zeta_potential'], p_cs_zeta, alpha=0.7, color='#1ABC9C', edgecolors='k', s=60)
    l_zeta = [5, 35]
    axes_cs[2].plot(l_zeta, l_zeta, 'r--', linewidth=2, label='Identity Line')
    axes_cs[2].set_xlabel("Experimental Zeta Potential (mV)", fontweight='bold')
    axes_cs[2].set_ylabel("Predicted Zeta Potential (mV)", fontweight='bold')
    axes_cs[2].set_title(f"Chitosan Zeta Potential (R2 = {r2_score(df_cs['zeta_potential'], p_cs_zeta):.3f})", fontweight='bold')
    axes_cs[2].legend()

    plt.tight_layout()
    fig_cs.savefig(os.path.join(benchmarks_dir, "chitosan_parity_plots.png"), dpi=300)
    fig_cs.savefig(os.path.join(benchmarks_dir, "chitosan_parity_plots.pdf"))
    plt.close()

    print("\n[SUCCESS] Model Training, Benchmarking, and Figure Generation Complete!")
    return models_bundle


if __name__ == "__main__":
    train_and_save_all()
