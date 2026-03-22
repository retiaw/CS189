import numpy as np
from CS189.lib.tensor import to_tensor
import os.path as osp
import json


def data_nan_process(X, nan_policy='mean', new_values=None):
    mask = ~np.isnan(X)
    if new_values is None:
        new_values = {}
    if nan_policy == 'mean':
        new_values['mean'] = np.nanmean(X, axis=0, keepdims=True)
        X = np.where(mask, X, new_values['mean'])
    elif nan_policy == 'median':
        new_values['median'] = np.nanmedian(X, axis=0, keepdims=True)
        X = np.where(mask, X, new_values['median'])
    else:
        raise ValueError(f'Unsupported nan policy: {nan_policy}')
    return X, new_values


def data_norm_process(X, norm_policy='mean_std', normalizer=None):
    if normalizer is None:
        normalizer = {}
    if norm_policy == 'mean_std':
        normalizer['mean'] = np.mean(X, axis=0, keepdims=True)
        normalizer['std'] = np.std(X, axis=0, keepdims=True)
        X = (X - normalizer['mean']) / (normalizer['std'] + 1e-6)
    elif norm_policy == 'min_max':
        normalizer['min'] = np.min(X, axis=0, keepdims=True)
        normalizer['max'] = np.max(X, axis=0, keepdims=True)
        X = (X - normalizer['min']) / (normalizer['max'] - normalizer['min'])
    else:
        raise ValueError(f'Unsupported norm policy: {norm_policy}')
    return X, normalizer


def label_process(y, y_policy='mean_std', is_regression=False,label_encoder=None):
    y = y.astype(float).reshape(-1) # y -> [b,]

    if label_encoder is None:
        label_encoder = {}
    if y_policy == 'mean_std':
        label_encoder['mean'] = np.mean(y)
        label_encoder['std'] = np.std(y)
        y = (y - label_encoder['mean']) / (label_encoder['std'] + 1e-6)
    elif y_policy == 'min_max':
        label_encoder['min'] = np.min(y)
        label_encoder['max'] = np.max(y)
        y = (y - label_encoder['min']) / (label_encoder['max'] - label_encoder['min'])
    elif y_policy == 'none':
        pass
    else:
        raise ValueError(f'Unsupported y policy: {y_policy}')
    return y, label_encoder


def data_loader_process(X, y, batch_size, shuffle=False, seed=None, device='cpu'):
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
        indices = rng.permutation(y.shape[0])
        X, y = X[indices], y[indices]

    def bin(data, batch_size):
        out = []
        for i in range(data.shape[0]):
            _tmp = []
            _str, _end = i, min(i+batch_size, data.shape[0])
            for j in range(_str, _end):
                _tmp.append(data[j])
            out.append(np.array(_tmp))
        return out

    X, y = bin(X, batch_size), bin(y, batch_size)
    
    def load(X, y):
        out = []
        for i in range(X.shape[0]):
            X[i] = to_tensor(X[i])
            y[i] = to_tensor(y[i])
            out.append((X[i], y[i]))
        return out

    loader = load(X, y)
    return loader


class Data:
    def __init__(self, data_path, data_name):
        self.data_path = data_path
        self.data_name = data_name

    def get_data_from_TALENT(self):
        _split = ['train', 'test']
        _dir = osp.join(self.data_path, self.data_name)
        if not osp.exists(_dir):
            raise FileNotFoundError(f'Directory {_dir} not found')

        def load_data(type):
            out = {}
            for s in _split:
                _path = osp.join(_dir, f'{type}_{s}.npy')
                if not osp.exists(_path):
                    out[s] = None
                else:
                    out[s] = np.load(_path, allow_pickle=True)
            return out

        N, y = load_data('N'), load_data('y')
        train_data = (N['train'], y['train'])
        test_data = (N['test'], y['test'])
        
        with open(osp.join(_dir, 'info.json'), 'r') as f:
            info = json.load(f)
        
        return train_data, test_data, info

    def get_data_from_CS189(self):
        # TODO: implement this
        pass