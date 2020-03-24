import random, logging, argparse, torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from models import BayesianCNN
from torch.utils.data.sampler import SubsetRandomSampler
from sampling import balanced_sample, StratifiedSampler
from active_learning import *

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


def train(args, model, train_loader, optimizer):
    model.train()
    train_set_size = len(train_loader) * train_loader.batch_size
    train_loss = 0
    correct = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # aggregate loss and corrects
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    ave_train_loss = train_loss / train_set_size
    proportion_correct = correct / train_set_size
    return train_loss, proportion_correct

def test(args, model, test_loader):
    model.eval()
    test_set_size = len(test_loader) * test_loader.batch_size
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    ave_test_loss = test_loss / test_set_size
    proportion_correct = correct / test_set_size
    return ave_test_loss, proportion_correct

def fit(model, optimizer, train_loader, test_loader, writers, args, i_round):
    writer1, writer2 = writers
    cumulative_acqs = args.acqs_pretrain + i_round * args.acqs_per_round
    max_test_correct = 0
    logging.info('Begin train/test for {} acquisitions'.format(cumulative_acqs)
    for epoch in range(1, args.epochs + 1): # TODO may to increase epochs for training on e.g. 1000 points to ensure convergence
        logging.info('Begin training, epoch {}'.format(epoch))
        train_loss, train_correct = train(args, model, train_loader, optimizer)
        cumulative_epoch = epoch + i_round * args.epochs
        writer1.add_scalar('loss', train_loss, cumulative_epoch)
        writer1.add_scalar('correct', train_correct, cumulative_epoch)
        # test on test set
        if epoch % args.test_interval == 0:
            logging.info('Begin testing, epoch {}'.format(epoch))
            test_loss, test_correct = test(args, model, test_loader)
            writer2.add_scalar('loss', test_loss, cumulative_epoch)
            writer2.add_scalar('correct', test_correct, cumulative_epoch)
            max_test_correct = max(test_correct, max_test_correct)
    # record max_test_correct
    writer2.add_scalar('max_test_correct', max_test_correct, cumulative_acqs)


def main():
    # experiment settings via command line
    args = parse_arguments()
    # setup logging
    args.logdir = './logs/{}/{}'.format(args.acq_func_ID, args.seed)
    logging.basicConfig(level=logging.INFO)
    # setup TensorBoard logging
    writer1 = SummaryWriter(log_dir=args.logdir+'-1')
    writer2 = SummaryWriter(log_dir=args.logdir+'-2')
    writers = [writer1, writer2]

    # for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load data
    train_data, test_data = load_data(args)
    # train_data, pool_idx, pretrain_loader, valid_loader, test_loader = load_data(args)
    # get idx of pretrain and validation data from train data
    pool_idx = set(range(len(train_data)))
    pretrain_idx, pool_idx = balanced_sample(train_data, n_classes=10, k=2, 
        idx_possible=pool_idx)
    valid_idx, pool_idx = balanced_sample(train_data, n_classes=10, k=10, 
        idx_possible=pool_idx)
    acq_idx = pretrain_idx.copy() # first 20 acquisitions are the pretraining data
    
    assert len(pretrain_idx) == 20
    assert len(valid_idx) == 100
    assert len(pool_idx) == len(train_data) - 120

    # make dataloaders
    # pretrain_sampler = SubsetRandomSampler(pretrain_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)
    # pretrain_loader = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.pretrain_batch_size, sampler=pretrain_sampler)
    # valid_loader = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.valid_batch_size, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.test_batch_size) # pytorch MNIST example shuffles test set, but seems unnecessary
    
    # pretraining
    # repeat for various choices of lambda:
    #    initial training on 20 points (random but *balanced*)
    #    compute validation error on 100 points
    # select lamba/model with lowest validation error
    # weight_decay = compute_weight_decay(pretrain_loader, valid_loader, args, writers)
    weight_decay = 1.0 # for now, let's not bother optimizing weight decay
    model = BayesianCNN()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    train_loader = get_train_loader(train_data, acq_idx, args)
    fit(model, optimizer, train_loader, test_loader, writers, args, i_round=0) # do pretraining on 20 examples (not quite clear if Gal does this here, but I think so)
    
    for i_round in range(1, args.rounds + 1):
        # acquire 10 points from train_data according to acq_func
        new_idx = args.acq_func(train_data, pool_idx, model, args)
        acq_idx.update(new_idx)
        pool_idx.difference_update(new_idx)
        # reinitalise model
        model = BayesianCNN()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        # train model to convergence on all points acquired so far, computing test error as we go
        # train_loader built from train_data and acq_idx, including pretraining examples
        train_loader = get_train_loader(train_data, acq_idx, args)
        fit(model, optimizer, train_loader, test_loader, writers, args, i_round)


if __name__ == '__main__':
    main()