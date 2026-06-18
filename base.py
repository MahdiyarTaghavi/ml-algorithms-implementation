from abc import ABC, abstractmethod


class BaseEstimator(ABC):

    @abstractmethod
    def fit(self, X, y, *args, **kwargs):
        pass

    def get_params(self, deep=True):
        params = {}

        for key, value in self.__dict__.items():

            # skip learned attributes
            if key.endswith("_"):
                continue

            # recurse only if deep + nested estimator
            if deep and hasattr(value, "get_params"):
                sub_params = value.get_params(deep=True)

                for sub_key, sub_value in sub_params.items():
                    params[f"{key}__{sub_key}"] = sub_value
            else:
                params[key] = value

        return params

    def set_params(self, **params):
        for key, value in params.items():

            parts = key.split("__")
            obj = self

            for p in parts[:-1]:
                obj = getattr(obj, p)

            setattr(obj, parts[-1], value)

        return self

