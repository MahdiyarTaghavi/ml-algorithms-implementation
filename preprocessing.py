from itertools import combinations_with_replacement

import numpy as np

from base import BaseEstimator
from mixins import TransformerMixin

class PolynomialFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias  # whether to add a column of 1s

    def fit(self, X, y=None):
        n_features = X.shape[1]
        self.n_features_in_ = n_features

        # Precompute which column indices to multiply for each output feature
        # e.g. (0,0) means x0*x0, (0,1) means x0*x1
        self.combinations_ = self._combinations(n_features)
        return self

    def transform(self, X):
        cols = []
        for combo in self.combinations_:
            if len(combo) == 0:                   # bias term: column of 1s
                cols.append(np.ones(X.shape[0]))
            else:
                cols.append(np.prod(X[:, combo], axis=1))
        return np.column_stack(cols)

    def _combinations(self, n_features):
        combos = []
        start = 0  # degree 0 = bias
        if not self.include_bias:
            start = 1
        for d in range(start, self.degree + 1):
            combos.extend(combinations_with_replacement(range(n_features), d))
        return combos