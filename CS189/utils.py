import numpy as np
import argparse
import json


def accuracy(y_hat, y):
    return np.sum(y_hat == y) / len(y)


def auc_roc(y_hat, y):
    threshold = np.arange(0, 1, 0.01)

    def FPR(tho):
        _y_hat = [y_hat >= tho]
        FP = np.sum([y==0 and _y_hat==1])
        TN = np.sum([y==0 and _y_hat==0])
        return FP / (FP + TN)

    def TPR(tho):
        _y_hat = [y_hat >= tho]
        TP = np.sum([y==1 and _y_hat==1])
        FN = np.sum([y==1 and _y_hat==0])
        return TP / (TP + FN)

    fpr, tpr = [], []
    for tho in threshold:
        fpr.append(FPR(tho))
        tpr.append(TPR(tho))

    indices = np.argsort(fpr)
    fpr = fpr[indices]
    tpr = tpr[indices]

    area, _pre_fpr, _pre_tpr = 0, 0, 0
    for i in len(fpr):
        area += (fpr[i] - _pre_fpr) * (tpr[i] + _pre_tpr) / 2
        _pre_fpr, _pre_tpr = fpr[i], tpr[i]
    
    return area


def mean_squared_error(y_hat, y, sqrt=False):
    mse = np.pow(np.linalg.norm(y_hat-y, ord=2), 2) / len(y)
    if sqrt is True:
        mse = np.sqrt(mse)
    return mse


def show_results(vres, metric_names):
    for name, value in zip(metric_names, vres):
        print(f'{name}: {value}')


def get_args():
    parser = argparse.ArgumentParser()
    with open('CS189/configs/default.json', 'r') as f:
        config = json.load(f)
    parser.add_argument('--data_path', type=str, default=config['data_path'])
    parser.add_argument('--data_name', type=str, default=config['data_name'])
    parser.add_argument('--model_type', type=str, default=config['model_type'])
    args = parser.parse_args()
    return args


def get_method(model_type: str):
    if model_type == 'my_linear_reg':
        from CS189.methods.linear_reg import MyLinearRegressionMethod
        return MyLinearRegressionMethod
    elif model_type == 'sk_linear_reg':
        from CS189.methods.linear_reg import SKLearnLinearRegressionMethod
        return SKLearnLinearRegressionMethod
    else:
        raise ValueError('Unsupported model type: {model_type}\n \
                          Supported model types: my_linear_reg, sk_linear_reg')




