"""Helper classes and functions for loading and sampling from data.
"""

import random, logging, argparse, torch
import numpy as np
import torch.optim as optim
from models import BayesianCNN
from torchvision import datasets, transforms
from torch.utils.data.sampler import Sampler, SubsetRandomSampler

class SubsetSampler(Sampler):
    """
    Samples elements from a given list of indices, without replacement.
    Had to write this function because it's not implemented in the library
    (see stackoverflow.com/questions/47432168/taking-subsets-of-a-pytorch-dataset)

    Parameters
    ----------
    indices: sequence
        sequence of indices defining the subset
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


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
        _, label, _ = dataset[i]
        if label_counts[label] < k:
            idx_sampled.add(i)
            idx_possible.remove(i)
            label_counts[label] += 1
            labels.append(label)
    assert all(labels.count(label) == k for label in range(n_classes))
    assert all(j not in idx_possible for j in idx_sampled)
    return idx_sampled, idx_possible


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


def load_data(args):
    MNISTWithIndices = dataset_with_indices(datasets.MNIST)
    train_data = MNISTWithIndices('../data', train=True, download=True,
                transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    test_data = MNISTWithIndices('../data', train=False, download=True,
                 transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    return train_data, test_data


def make_dataloader(data, idx, batch_size, random=True):
    """
    Parameters
    ----------
    data: torch.utils.data.Dataset
        base dataset to be sampled from
    idx: set
        indices of data that dataloader will load data from
    args: Namespace object
        experiment arguments from argparse
    
    Returns
    -------
    dataloader: torch.utils.data.DataLoader
        DataLoader iterable for iterating through train/test data
    """
    if random:
        sampler = SubsetRandomSampler(list(idx))
    else:
        sampler = SubsetSampler(list(idx))
    dataloader = torch.utils.data.DataLoader(
                    data, batch_size=batch_size, sampler=sampler)
    return dataloader


# def compute_weight_decay(pretrain_loader, valid_loader, args, writers):
#     """
#     From the paper: "All models are trained on the MNIST dataset with a (random
#     but balanced) initial training set of 20 data points, and a validation set
#     of 100 points on which we optimise the weight decay."
#     """
#     corrects = []
#     writer1, writer2 = writers
#     valid_set_size = len(valid_loader) * valid_loader.batch_size
#     weight_decays = [1, 1e-1, 1e-2, 1e-3, 1e-4]
#     for weight_decay in weight_decays:
#         logging.info('Pretraining with weight decay {}'.format(weight_decay))
#         max_correct = 0
#         model = BayesianCNN()
#         optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay) # NB Gal doesn't mention which optimizer they use
#         for epoch in range(50):
#             pretrain_loss = train(args, model, pretrain_loader, optimizer)
#             writer1.add_scalar('pretrain_loss_{}'.format(weight_decay), pretrain_loss, epoch)
#             # test on validation set
#             valid_loss, correct = test(args, model, valid_loader)
#             writer2.add_scalar('pretrain_loss_{}'.format(weight_decay), valid_loss, epoch)
#             writer2.add_scalar('pretrain_correct_{}'.format(weight_decay), correct, epoch)
#             max_correct = max(correct, max_correct)
#         # record max correct
#         corrects.append(max_correct)
#         logging.info('\nValidation set: Accuracy: {}/{} ({:.0f}%)\n'.format(
#                      max_correct, valid_set_size,
#                      100. * max_correct / valid_set_size))
#     return weight_decays[np.argmax(np.array(corrects))]