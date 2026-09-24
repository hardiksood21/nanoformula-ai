"""
Applicability Domain (AD) module for Nanoparticle Formulation Models.
Implements Leverage (William's Plot) and Mahalanobis / Euclidean distance bounding
to assess whether a new formulation falls within the reliable interpolation domain.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union


class ApplicabilityDomain:
    """
    Assesses whether a query formulation falls within the training domain
    using leverage (Hat matrix diagonal) and Mahalanobis distance metrics.
    """
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.mean_ = None
        self.std_ = None
        self.inv_cov_ = None
        self.X_train_std_ = None
        self.leverage_threshold_ = None
        self.distance_threshold_95_ = None
        self.n_samples_ = 0
        self.n_features_ = 0

    def fit(self, X: Union[pd.DataFrame, np.ndarray]):
        """
        Fits the applicability domain on training feature matrix.
        """
        if isinstance(X, pd.DataFrame):
            X_mat = X[self.feature_names].values.astype(float)
        else:
            X_mat = np.array(X, dtype=float)

        self.n_samples_, self.n_features_ = X_mat.shape
        self.mean_ = np.mean(X_mat, axis=0)
        self.std_ = np.std(X_mat, axis=0)
        self.std_[self.std_ == 0] = 1.0  # Avoid division by zero

        # Standardized features
        self.X_train_std_ = (X_mat - self.mean_) / self.std_

        # Regularized pseudo-inverse covariance for Mahalanobis
        cov = np.cov(self.X_train_std_, rowvar=False)
        cov += np.eye(self.n_features_) * 1e-5  # Ridge regularization
        self.inv_cov_ = np.linalg.pinv(cov)

        # William's leverage warning threshold: h* = 3*(p + 1)/n
        p = self.n_features_
        n = self.n_samples_
        self.leverage_threshold_ = 3.0 * (p + 1) / n

        # Training set Mahalanobis distances
        train_dists = [self._calc_mahalanobis(x) for x in self.X_train_std_]
        self.distance_threshold_95_ = np.percentile(train_dists, 95)
        return self

    def _calc_mahalanobis(self, x_std: np.ndarray) -> float:
        diff = x_std
        return float(np.sqrt(np.dot(np.dot(diff, self.inv_cov_), diff.T)))

    def _calc_leverage(self, x_std: np.ndarray) -> float:
        # Approximate leverage via projection
        try:
            pinv_XTX = np.linalg.pinv(np.dot(self.X_train_std_.T, self.X_train_std_))
            h = float(np.dot(np.dot(x_std, pinv_XTX), x_std.T))
            return max(0.0, h)
        except Exception:
            return 0.0

    def check(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Checks whether query points are within domain.
        Returns AD status, leverage value, distance, and reliability score (0-100%).
        """
        if isinstance(X, dict):
            X_mat = np.array([[X[f] for f in self.feature_names]], dtype=float)
        elif isinstance(X, pd.DataFrame):
            X_mat = X[self.feature_names].values.astype(float)
        else:
            X_mat = np.array(X, dtype=float)
            if X_mat.ndim == 1:
                X_mat = X_mat.reshape(1, -1)

        X_std = (X_mat - self.mean_) / self.std_
        results = []

        for i in range(len(X_std)):
            x_i = X_std[i]
            dist = self._calc_mahalanobis(x_i)
            leverage = self._calc_leverage(x_i)
            
            in_leverage = leverage <= self.leverage_threshold_
            in_distance = dist <= self.distance_threshold_95_

            if in_leverage and in_distance:
                status = "In Domain (High Reliability)"
                level = "HIGH"
                reliability_score = max(80.0, 100.0 - (dist / self.distance_threshold_95_) * 20.0)
            elif in_leverage or in_distance:
                status = "Borderline (Moderate Reliability)"
                level = "MODERATE"
                reliability_score = max(50.0, 80.0 - (dist / (self.distance_threshold_95_ * 1.5)) * 30.0)
            else:
                status = "Out of Domain (Extrapolation Risk)"
                level = "LOW"
                reliability_score = max(10.0, 50.0 - (dist / (self.distance_threshold_95_ * 2.0)) * 40.0)

            results.append({
                "status": status,
                "confidence_level": level,
                "reliability_score": round(float(reliability_score), 1),
                "leverage": round(float(leverage), 4),
                "leverage_threshold": round(float(self.leverage_threshold_), 4),
                "mahalanobis_distance": round(float(dist), 2),
                "distance_threshold": round(float(self.distance_threshold_95_), 2),
                "in_leverage": in_leverage,
                "in_distance": in_distance
            })

        if len(results) == 1:
            return results[0]
        return {"samples": results}
