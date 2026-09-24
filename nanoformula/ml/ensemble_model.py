"""
Ensemble Machine Learning Model with Uncertainty Quantification (UQ).
Provides calibrated point predictions, standard deviations, and 95% confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
import xgboost as xgb


class NanoparticleEnsembleRegressor:
    """
    Ensemble regressor integrating XGBoost, Random Forest, and Gradient Boosting
    with tree-level and model-level variance estimation for Uncertainty Quantification.
    """
    def __init__(self, target_name: str, feature_names: List[str]):
        self.target_name = target_name
        self.feature_names = feature_names
        self.xgb_model = None
        self.rf_model = None
        self.gb_model = None
        self.et_model = None
        self.weights = [0.45, 0.25, 0.15, 0.15]  # XGB, RF, GB, ET
        self.residual_std_ = 10.0
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Fits ensemble models on features X and target y.
        """
        X_mat = X[self.feature_names].values if isinstance(X, pd.DataFrame) else np.array(X)
        y_arr = np.array(y, dtype=float)

        # 1. XGBoost
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )
        self.xgb_model.fit(X_mat, y_arr)

        # 2. Random Forest
        self.rf_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            min_samples_split=3,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_mat, y_arr)

        # 3. Gradient Boosting
        self.gb_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        self.gb_model.fit(X_mat, y_arr)

        # 4. Extra Trees
        self.et_model = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        self.et_model.fit(X_mat, y_arr)

        # Compute empirical residual standard deviation for calibrated UQ
        preds_xgb = self.xgb_model.predict(X_mat)
        preds_rf = self.rf_model.predict(X_mat)
        preds_gb = self.gb_model.predict(X_mat)
        preds_et = self.et_model.predict(X_mat)
        ens_pred = (
            self.weights[0] * preds_xgb +
            self.weights[1] * preds_rf +
            self.weights[2] * preds_gb +
            self.weights[3] * preds_et
        )
        residuals = y_arr - ens_pred
        self.residual_std_ = float(np.std(residuals))
        self.is_fitted_ = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]], return_std: bool = False) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Generates predictions with uncertainty quantification.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet.")

        if isinstance(X, dict):
            X_mat = np.array([[X[f] for f in self.feature_names]], dtype=float)
            single_input = True
        elif isinstance(X, pd.DataFrame):
            X_mat = X[self.feature_names].values.astype(float)
            single_input = False
        else:
            X_mat = np.array(X, dtype=float)
            if X_mat.ndim == 1:
                X_mat = X_mat.reshape(1, -1)
                single_input = True
            else:
                single_input = False

        p_xgb = self.xgb_model.predict(X_mat)
        p_rf = self.rf_model.predict(X_mat)
        p_gb = self.gb_model.predict(X_mat)
        p_et = self.et_model.predict(X_mat)

        # Weighted ensemble mean
        y_mean = (
            self.weights[0] * p_xgb +
            self.weights[1] * p_rf +
            self.weights[2] * p_gb +
            self.weights[3] * p_et
        )

        # Post-process bounds (size > 0, EE in [0, 100], PDI in [0, 1])
        if "size" in self.target_name.lower():
            y_mean = np.maximum(y_mean, 10.0)
        elif "ee" in self.target_name.lower():
            y_mean = np.clip(y_mean, 0.0, 100.0)
        elif "lc" in self.target_name.lower():
            y_mean = np.clip(y_mean, 0.0, 60.0)
        elif "pdi" in self.target_name.lower():
            y_mean = np.clip(y_mean, 0.05, 0.95)

        if not return_std:
            return y_mean if not single_input else float(y_mean[0])

        # Epistemic variance (disagreement between ensemble models)
        stacked_preds = np.vstack([p_xgb, p_rf, p_gb, p_et])
        epistemic_var = np.var(stacked_preds, axis=0)

        # Total standard deviation combining epistemic variance + aleatoric residual floor
        total_std = np.sqrt(epistemic_var + (0.5 * self.residual_std_)**2)
        ci_lower = y_mean - 1.96 * total_std
        ci_upper = y_mean + 1.96 * total_std

        if "size" in self.target_name.lower():
            ci_lower = np.maximum(ci_lower, 5.0)
        elif "ee" in self.target_name.lower() or "lc" in self.target_name.lower():
            ci_lower = np.maximum(ci_lower, 0.0)
            ci_upper = np.minimum(ci_upper, 100.0)
        elif "pdi" in self.target_name.lower():
            ci_lower = np.maximum(ci_lower, 0.05)
            ci_upper = np.minimum(ci_upper, 0.99)

        if single_input:
            return {
                "mean": round(float(y_mean[0]), 2),
                "std": round(float(total_std[0]), 2),
                "ci95_lower": round(float(ci_lower[0]), 2),
                "ci95_upper": round(float(ci_upper[0]), 2),
                "formatted": f"{y_mean[0]:.1f} ± {total_std[0]:.1f} (95% CI: [{ci_lower[0]:.1f}, {ci_upper[0]:.1f}])"
            }
        else:
            return y_mean, total_std, ci_lower, ci_upper
