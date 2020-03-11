"""
Classes for implementing stratified sampling (random but balanced).
"""

import torch, random

def balanced_sample(dataset, n_classes, k, idx_possible):
    """
    Returns indices `idx_sampled` of `n_classes` * `k` random samples from dataset, s.t. each
    class is sampled `k` times. Also returns remaining indices (i.e.
    `idx_possible` - `idx_sampled`).
    
    Parameters
    ----------
    dataset: torchvision.datasets.mnist.MNIST
        a PyTorch dataset
    n_classes: int
        number of class labels in `dataset`
    k: int
        required number of samples of each class in our balanced sample
    idx: list
        list of indices in dataset that we can sample from
    
    Returns
    -------
    TODO complete docstr
    """
    idx_sampled = set()
    labels = [] # used to check for correctness
    label_counts = {label: 0 for label in range(n_classes)}
    while any(count < k for count in label_counts.values()):
        i = random.choice(tuple(idx_possible))
        data, label = dataset[i]
        if label_counts[label] < k:
            idx_sampled.add(i)
            idx_possible.remove(i)
            label_counts[label] += 1
            labels.append(label)
    assert all(labels.count(label) == k for label in range(n_classes))
    assert all(j not in idx_possible for j in idx_sampled)
    return idx_sampled, idx_possible


class Sampler(object):
    """
    Base class for all Samplers.
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    Copied from: github.com/ncullen93/torchsample/blob/master/torchsample/samplers.py
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class StratifiedSampler(Sampler):
    """
    Stratified Sampling
    Provides equal representation of target classes in each batch
    Copied from: github.com/ncullen93/torchsample/blob/master/torchsample/samplers.py
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np
        
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0),2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)