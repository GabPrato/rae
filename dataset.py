import h5py
import numpy as np
import random


class Dataset:
    def __init__(self, params):
        self.params = params
        dataset = h5py.File(params.dataset_path, 'r')
        self.embeddings = dataset['embeddings'][:]

        self.train = {}
        for d in dataset['train'].values():
            if d.shape[1] > params.max_sequence_len or d.shape[0] < params.batch_size:
                continue
            self.train[d.shape[1]] = d[:]

        self.validation = {}
        for d in dataset['validation'].values():
            if d.shape[1] > params.max_sequence_len:
                continue
            self.validation[d.shape[1]] = d[:]

        self.test = {}
        for d in dataset['test'].values():
            if d.shape[1] > params.max_sequence_len:
                continue
            self.test[d.shape[1]] = d[:]

        dataset.close()
        self.reset()

    def reset(self):
        self.indices_per_dataset = {}
        self.current_ids = {}
        self.dataset_iterations = []
        for k, d in self.train.items():
            ids = np.arange(d.shape[0])
            np.random.shuffle(ids)
            self.indices_per_dataset[k] = ids
            
            self.dataset_iterations += [k] * (d.shape[0] // self.params.batch_size)

            self.current_ids[k] = 0

        random.shuffle(self.dataset_iterations)

    def train_epoch(self):
        while len(self.dataset_iterations) != 0:
            i = self.dataset_iterations.pop()
            yield self.embeddings[self.train[i][self.indices_per_dataset[i][self.current_ids[i] : self.current_ids[i] + self.params.batch_size]]]
            self.current_ids[i] += self.params.batch_size

        self.reset()

    def test_epoch(self, val_or_test):
        _set = eval(f'self.{val_or_test}')
        for d in _set.values():
            for i in range(0, d.shape[0] - (d.shape[0] % self.params.batch_size), self.params.batch_size):
                yield self.embeddings[d[i:i+self.params.batch_size]]
