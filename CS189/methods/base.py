import numpy as np
from abc import ABC, abstractmethod
from CS189.utils import mean_squared_error, accuracy, auc_roc


class Method(ABC):
    def __init__(self, args, info):
        self.args = args
        self.info = info
    
    @abstractmethod
    def data_format(self, X, y, is_train=False):
        pass

    @abstractmethod
    def fit(self, train_data):
        pass

    @abstractmethod
    def predict(self, test_data):
        pass

    def metrics(self, pred, y):
        # TODO: need to implement
        if self.info['task_type'] == 'regression':
            if self.info['y_policy'] == 'mean_std':
                pred = pred * self.y_std + self.y_mean
                y = y * self.y_std + self.y_mean
            rmse = mean_squared_error(pred, y, sqrt=True)
            return (rmse,), ('RMSE',)
        else:
            _y_hat = (pred >= 0.5).astype(int)
            acc = accuracy(_y_hat, y)
            auc = auc_roc(pred, y)
            return acc, auc, 'Accuracy', 'AUC'