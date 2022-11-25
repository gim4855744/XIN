from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

class Preprocessor:
    
    def __init__(self, task) -> None:
        self._numerical_fields_transformer = MinMaxScaler(feature_range=(-1, 1))
        self._categorical_fields_transformer = OrdinalEncoder(dtype=int, handle_unknown='use_encoded_value', unknown_value=-1)
        if task == "regression":
            self._label_transformaer = MinMaxScaler(feature_range=(-1, 1))
        else:
            self._label_transformaer = OrdinalEncoder(dtype=int)

    def fit(self, numerical_fields, categorical_fields, label):
        self._numerical_fields_transformer.fit(numerical_fields)
        self._categorical_fields_transformer.fit(categorical_fields)
        self._label_transformaer.fit(label)

    def transform(self, numerical_fields, categorical_fields, label):

        transformed_numerical_fields = self._numerical_fields_transformer.transform(numerical_fields)
        transformed_categorical_fields = self._categorical_fields_transformer.transform(categorical_fields)
        transformed_label = self._label_transformaer.transform(label)

        # convert unknown value
        for i in range(self._categorical_fields_transformer.n_features_in_):
            x = transformed_categorical_fields[:, i]
            x[x == -1] = len(self._categorical_fields_transformer.categories_[i])

        return transformed_numerical_fields, transformed_categorical_fields, transformed_label

    def fit_transform(self, numerical_fields, categorical_fields, label):
        self.fit(numerical_fields, categorical_fields, label)
        return self.transform(numerical_fields, categorical_fields, label)
