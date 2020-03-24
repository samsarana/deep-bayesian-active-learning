import random, logging, argparse, torch
import numpy as np
import torch.optim as optim
from models import BayesianCNN
from active_learning import *
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--acqs_per_round', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=50) # Gal does 100
    parser.add_argument('--acqs_pretrain', type=int, default=20)
    # training settings
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for Adam optimizer')
    parser.add_argument('--pretrain_batch_size', type=int, default=20, help='Batch size for pretraining with 20 examples')
    parser.add_argument('--valid_batch_size', type=int, default=20, help='Batch size for pretraining validation with 100 examples')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training (on 10-1000 examples)')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Batch size for testing on 10k examples')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--test_interval', type=int, default=10, help='Number of epochs to train before testing')
    # active learning settings
    acq_funcs = {'random': acq_random,
                 'BALD': acq_BALD,
                 'max_ent': acq_max_ent,
                 'mean_std': acq_mean_std,
                 'var_ratios': acq_var_ratios
    }
    parser.add_argument('--acq_func_ID', type=str, default='random', choices=acq_funcs.keys(), help='Choose acquisition function')
    args = parser.parse_args()
    args.acq_func = acq_funcs[args.acq_func_ID] # create acq_func arg using acq_func_ID
    return args


def load_data(args):
    train_data = datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    test_data = datasets.MNIST('../data', train=False, download=True,
                 transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    return train_data, test_data


def get_train_loader(train_data, acq_idx, args):
    train_sampler = SubsetRandomSampler(list(acq_idx))
    train_loader = torch.utils.data.DataLoader(
                    train_data, batch_size=args.train_batch_size, sampler=train_sampler)
    return train_loader


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