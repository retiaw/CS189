import numpy as np
from abc import ABC, abstractmethod
from CS189.lib.data import data_nan_process, data_norm_process, label_process
from CS189.utils import mean_squared_error, accuracy, auc_roc


class Method(ABC):
    def __init__(self, args, info):
        self.args = args
        self.info = info
    
    def data_format(self, X, y, is_train=False):
        if is_train:
            X, self.new_values = data_nan_process(X, nan_policy=self.args.nan_policy, new_values=None)
            X, self.normalizer = data_norm_process(X, norm_policy=self.args.norm_policy, normalizer=None)
            y, self.label_encoder = label_process(y, y_policy=self.args.y_policy, is_regression=self.info['task_type']=='regression', label_encoder=None)
            self.info['d_in'] = X.shape[-1]
            if self.info['task_type'] != 'regression':
                self.info['n_classes'] = len(np.unique(y))
            return X, y
        else:
            X, _ = data_nan_process(X, self.args.nan_policy, self.new_values)
            X, _ = data_norm_process(X, self.args.norm_policy, self.normalizer)
            y, _ = label_process(y, self.args.y_policy, is_regression=self.info['task_type']=='regression', label_encoder=self.label_encoder)
            return X, y

    @abstractmethod
    def fit(self, train_data):
        pass

    @abstractmethod
    def predict(self, test_data):
        pass

    def metrics(self, y_hat, y):
        """
        regression:
        y_hat: [b,] predicted values
        y:     [b,] true values
        
        classification:
        y_hat: [b, c]   predicted probabilities
        y:     [b, ...] true labels

        """

        if self.info['task_type'] == 'regression':
            if self.args.y_policy == 'mean_std':
                y_hat = y_hat * self.label_encoder['std'] + self.label_encoder['mean']
                y = y * self.label_encoder['std'] + self.label_encoder['mean']
            rmse = mean_squared_error(y_hat, y, sqrt=True)
            return (rmse,), ('RMSE',)
        elif self.info['task_type'] == 'binclass':
            if self.args.y_policy == 'one_hot':
                y = np.argmax(y, axis=-1) # [b,]
            acc = accuracy(y_hat, y)
            auc = auc_roc(y_hat, y)
            return (acc, auc), ('Accuracy', 'AUC')
        else:
            if self.args.y_policy == 'one_hot':
                y = np.argmax(y, axis=-1) # [b,]
            acc = accuracy(y_hat, y)
            return (acc,), ('Accuracy',)