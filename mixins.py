class PredictorMixin:
    def predict(self, X):
        pass

class TransformerMixin:
    def transform(self, X):
        pass

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class RegressorMixin:
    def score(self, X, y):
        pass