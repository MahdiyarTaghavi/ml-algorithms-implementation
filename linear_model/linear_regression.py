from base import BaseEstimator
from mixins import PredictorMixin, RegressorMixin

class LinearRegression(BaseEstimator, PredictorMixin, RegressorMixin):

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        raise NotImplementedError("LinearRegression.fit is not implemented yet.")

    def predict(self, X):
        raise NotImplementedError("LinearRegression.predict is not implemented yet.")

