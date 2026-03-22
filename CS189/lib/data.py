import numpy as np
from CS189.lib.tensor import to_tensor
import os.path as osp
import json


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