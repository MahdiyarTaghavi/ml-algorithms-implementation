import numpy as np

from base import BaseEstimator
from mixins import PredictorMixin, RegressorMixin

class LinearRegression(BaseEstimator, PredictorMixin, RegressorMixin):

    def __init__(self, fit_intercept=True):
        self.fit_intercept: bool = fit_intercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Fit the model using Ordinary Least Squares via SVD.

        Solves β = argmin ||Xβ - y||² using np.linalg.lstsq which
        computes the pseudoinverse X⁺ = V Σ⁺ Uᵀ internally, giving
        β = X⁺y without ever inverting XᵀX directly.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        self
        """
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        if self.fit_intercept:
            self.intercept_ = coeffs[0]
            self.coef_ = coeffs[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = coeffs

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

