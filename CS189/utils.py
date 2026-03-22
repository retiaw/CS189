import numpy as np
import argparse
import json


def log_normal_distribution(X, mu, sigma):
    """
    X: [b, d]
    mu: [c, d]
    sigma: [c, d, d]
    """
    if sigma.ndim == 2: sigma = sigma[None,:,:]
    if mu.ndim == 1: mu = mu[None,:]

    dim = X.shape[-1]
    _, log_det = np.linalg.slogdet(sigma)
    alpha = -0.5 * (dim * np.log(2*np.pi) + log_det)
    x_centre = X[None,:,:] - mu[:,None,:] # [c, b, d]
    core = -0.5 * np.einsum('cbd,cdb->bc',
                            x_centre,
                            np.linalg.solve(sigma, x_centre.transpose(0, 2, 1)))
    fn = alpha.reshape(1, -1) + core
    return fn #[b, c]


def softmax(x):
    """
    x: [b, d]    
    """
    x = x - np.max(x, axis=-1, keepdims=True)
    x = np.exp(x)
    x = x / np.sum(x, axis=-1, keepdims=True)
    return x # [b, d]


def accuracy(y_hat, y):
    """
    y_hat: [b, c]
    y:     [b,]

    """
    y_hat = np.argmax(y_hat, axis=-1)
    return np.sum(y_hat == y) / y.shape[0]


def auc_roc(y_hat, y):
    """
    y_hat: [b, 2]
    y:     [b,]

    """
    positive_probs = y_hat[:, 1] # [b,]
    threshold = np.arange(0, 1, 0.01)

    def FPR(tho):
        _positive_probs = (positive_probs >= tho)
        FP = np.sum((y == 0) & (_positive_probs == 1))
        TN = np.sum((y == 0) & (_positive_probs == 0))
        return FP / (FP + TN)

    def TPR(tho):
        _positive_probs = (positive_probs >= tho)
        TP = np.sum((y == 1) & (_positive_probs == 1))
        FN = np.sum((y == 1) & (_positive_probs == 0))
        return TP / (TP + FN)

    fpr, tpr = [], []
    for tho in threshold:
        fpr.append(FPR(tho))
        tpr.append(TPR(tho))
    fpr, tpr = np.array(fpr), np.array(tpr)

    indices = np.argsort(fpr)
    fpr = fpr[indices]
    tpr = tpr[indices]

    area, _pre_fpr, _pre_tpr = 0, 0, 0
    for i in range(fpr.shape[0]):
        area += (fpr[i] - _pre_fpr) * (tpr[i] + _pre_tpr) / 2
        _pre_fpr, _pre_tpr = fpr[i], tpr[i]
    
    return area


def mean_squared_error(y_hat, y, sqrt=False):
    mse = np.pow(np.linalg.norm(y_hat-y, ord=2), 2) / y.shape[0]
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
    parser.add_argument('--nan_policy', type=str, default=config['nan_policy'])
    parser.add_argument('--norm_policy', type=str, default=config['norm_policy'])
    parser.add_argument('--y_policy', type=str, default=config['y_policy'])
    args = parser.parse_args()
    return args


def get_method(model_type: str):
    if model_type == 'my_linear_reg':
        from CS189.methods.linear_reg import MyLinearRegressionMethod
        return MyLinearRegressionMethod
    elif model_type == 'sk_linear_reg':
        from CS189.methods.linear_reg import SKLearnLinearRegressionMethod
        return SKLearnLinearRegressionMethod
    elif model_type == 'my_lda':
        from CS189.methods.gda import MyLDA_Method
        return MyLDA_Method
    elif model_type == 'sk_lda':
        from CS189.methods.gda import SK_LDA_Method
        return SK_LDA_Method
    elif model_type == 'my_qda':
        from CS189.methods.gda import MyQDA_Method
        return MyQDA_Method
    elif model_type == 'sk_qda':
        from CS189.methods.gda import SK_QDA_Method
        return SK_QDA_Method

    else:
        raise ValueError('Unsupported model type: {model_type}\n \
                          Supported model types: my_linear_reg, sk_linear_reg')




