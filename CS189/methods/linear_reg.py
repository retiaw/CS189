from CS189.methods.base import Method
import numpy as np


class LinearRegressionMethod(Method):
    def __init__(self, args, info):
        super().__init__(args, info)
        assert self.info['task_type'] == 'regression'

    def data_format(self, X, y, is_train=True):
        if y.ndim > 1:
            y = y.reshape(-1)

        if is_train:
            self.info['y_policy'] = 'mean_std'
            self.X_mean, self.X_std = np.mean(X, axis=0).reshape(1, -1), np.std(X, axis=0).reshape(1, -1)
            self.y_mean, self.y_std = np.mean(y), np.std(y)
        
        X = (X - self.X_mean) / self.X_std
        y = (y - self.y_mean) / self.y_std

        return X, y
    

class MyLinearRegressionMethod(LinearRegressionMethod):
    def __init__(self, args, info):
        super().__init__(args, info)
        
    def fit(self, train_data):
        X, y = train_data[0], train_data[1]
        X, y = self.data_format(X, y, is_train=True)
        X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=float)], axis=1)
        _X = X.T @ X
        _y = X.T @ y
        self.w = np.linalg.solve(_X, _y)

    def predict(self, test_data):
        X_test, y_test = test_data[0], test_data[1]
        X_test, y_test = self.data_format(X_test, y_test, is_train=False)
        X_test = np.concatenate([X_test, np.ones((X_test.shape[0], 1), dtype=float)], axis=1)
        pred = np.einsum('ni,i->n', X_test, self.w).squeeze()
        vres, metric_names = self.metrics(pred, y_test)
        return vres, metric_names, pred
    

class SKLearnLinearRegressionMethod(LinearRegressionMethod):
    def __init__(self, args, info):
        super().__init__(args, info)
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression(fit_intercept=True)

    def fit(self, train_data):
        X, y = train_data[0], train_data[1]
        X, y = self.data_format(X, y, is_train=True)
        self.model.fit(X, y)

    def predict(self, test_data):
        X_test, y_test = test_data[0], test_data[1]
        X_test, y_test = self.data_format(X_test, y_test, is_train=False)
        pred = self.model.predict(X_test)
        vres, metric_names = self.metrics(pred, y_test)
        return vres, metric_names, pred