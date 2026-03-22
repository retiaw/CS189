import numpy as np
from CS189.utils import log_normal_distribution, softmax
from CS189.methods.base import Method
from CS189.lib.data import data_nan_process


class GDA_Method(Method):
    def __init__(self, args, info):
        super().__init__(args, info)
        assert self.info['task_type'] != 'regression'
    

class SK_LDA_Method(GDA_Method):
    def __init__(self, args, info):
        super().__init__(args, info)
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        self.model = LinearDiscriminantAnalysis(solver='lsqr',
                                                shrinkage=None,
                                                store_covariance=True)
        
    def fit(self, train_data):
        X, y = train_data[0], train_data[1]
        X, y = self.data_format(X, y, is_train=True)
        self.model.fit(X, y)
    
    def predict(self, test_data):
        X_test, y_test = test_data[0], test_data[1]
        X_test, y_test = self.data_format(X_test, y_test, is_train=False)
        probs = self.model.predict_proba(X_test) # [b, c]
        vres, metric_names = self.metrics(probs, y_test)
        pred = np.argmax(probs, axis=-1) # [b,]
        return vres, metric_names, pred
    

class MyLDA_Method(GDA_Method):
    def __init__(self, args, info):
        super().__init__(args, info)
    
    def fit(self, train_data):
        X, y = train_data[0], train_data[1]
        X, y = self.data_format(X, y, is_train=True)
        _classes, _classes_counts = np.unique(y, return_counts=True)
        
        self.prior = _classes_counts / self.info['n_classes']
        self.mu = np.empty((self.info['n_classes'], self.info['d_in'])) # [c, d]
        self.sigma = np.empty((self.info['n_classes'], self.info['d_in'], self.info['d_in'])) # [c, d, d]

        for i, c in enumerate(_classes):
            mask = (y == c)
            self.mu[i] = np.mean(X[mask], axis=0)
            _x_centre = X[mask] - self.mu[i]
            self.sigma[i] = np.einsum('bd,bi->di', _x_centre, _x_centre) / np.sum(mask)

        self.sigma = np.squeeze(np.einsum('ci,cdj->idj', self.prior.reshape(-1, 1), self.sigma), axis=0)
        lamda, _ = np.linalg.eigh(self.sigma)
        lamda = np.min(lamda)
        if lamda < 0:
            self.sigma = self.sigma - np.eye(self.info['d_in'], self.info['d_in']) * lamda
    
    def predict(self, test_data):
        X_test, y_test = test_data[0], test_data[1]
        X_test, y_test = self.data_format(X_test, y_test, is_train=False)
        q = np.log(self.prior[None,:]) + log_normal_distribution(X_test, self.mu, self.sigma)
        probs = softmax(q) # [b, c]
        vres, metric_names = self.metrics(probs, y_test)
        pred = np.argmax(probs, axis=-1) # [b,]
        return vres, metric_names, pred


class SK_QDA_Method(GDA_Method):
    def __init__(self, args, info):
        super().__init__(args, info)
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        self.model = QuadraticDiscriminantAnalysis(priors=None, reg_param=1e-3)

    def fit(self, train_data):
        X, y = train_data[0], train_data[1]
        X, y = self.data_format(X, y, is_train=True)
        self.model.fit(X, y)
    
    def predict(self, test_data):
        X_test, y_test = test_data[0], test_data[1]
        X_test, y_test = self.data_format(X_test, y_test, is_train=False)
        probs = self.model.predict_proba(X_test) # [b, c]
        vres, metric_names = self.metrics(probs, y_test)
        pred = np.argmax(probs, axis=-1) # [b,]
        return vres, metric_names, pred


class MyQDA_Method(GDA_Method):
    def __init__(self, args, info):
        super().__init__(args, info)
        self.lamda = 1e-3

    def fit(self, train_data):
        X, y = train_data[0], train_data[1]
        X, y = self.data_format(X, y, is_train=True)
        _classes, _classes_counts = np.unique(y, return_counts=True)
        
        self.prior = _classes_counts / self.info['n_classes']
        self.mu = np.empty((self.info['n_classes'], self.info['d_in'])) # [c, d]
        self.sigma = np.empty((self.info['n_classes'], self.info['d_in'], self.info['d_in'])) # [c, d, d]

        for i, c in enumerate(_classes):
            mask = (y == c)
            self.mu[i] = np.mean(X[mask], axis=0)
            _x_centre = X[mask] - self.mu[i]
            self.sigma[i] = np.einsum('bd,bi->di', _x_centre, _x_centre) / np.sum(mask)
            self.sigma[i] = (1 - self.lamda) * self.sigma[i] + self.lamda * np.eye(self.info['d_in'], self.info['d_in'])            

    def predict(self, test_data):
        X_test, y_test = test_data[0], test_data[1]
        X_test, y_test = self.data_format(X_test, y_test, is_train=False)
        q = np.log(self.prior[None,:]) + log_normal_distribution(X_test, self.mu, self.sigma)
        probs = softmax(q) # [b, c]
        vres, metric_names = self.metrics(probs, y_test)
        pred = np.argmax(probs, axis=-1) # [b,]
        return vres, metric_names, pred