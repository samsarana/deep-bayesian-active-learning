import random, logging, argparse, torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from models import BayesianCNN
from torch.utils.data.sampler import SubsetRandomSampler
from sampling import balanced_sample, StratifiedSampler

def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--seed', type=int, default=0)
    # training settings
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for Adam optimizer')
    parser.add_argument('--pretrain_batch_size', type=int, default=20, help='Batch size for pretraining with 20 examples')
    parser.add_argument('--valid_batch_size', type=int, default=20, help='Batch size for pretraining validation with 100 examples')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training (on 10-1000 examples)')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Batch size for testing on 10k examples')
    # active learning settings
    parser.add_argument('--acq_func_ID', type=str, default='random', help='Choice of random, bald, max_ent, mean_std or var_ratios acquisition functions')
    args = parser.parse_args()
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

    # return train_data, set(pool_idx), pretrain_loader, valid_loader, test_loader


def train(args, model, train_loader, optimizer):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    return loss

def test(args, model, test_loader):
    test_set_size = len(test_loader) * test_loader.batch_size
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= test_set_size
    return test_loss, correct
    

def get_train_loader(train_data, acq_idx, args):
    train_sampler = SubsetRandomSampler(list(acq_idx))
    train_loader = torch.utils.data.DataLoader(
                    train_data, batch_size=args.train_batch_size, sampler=train_sampler)
    return train_loader

def fit(model, optimizer, train_loader, test_loader, writers, args, n_acqs, epochs):
    writer1, writer2 = writers
    max_correct = 0
    for epoch in range(epochs): # TODO may to increase epochs for training on 1000 points to ensure convergence
        train_loss = train(args, model, train_loader, optimizer)
        writer1.add_scalar('train_loss_{}'.format(n_acqs), train_loss, epoch)
        # test on test set
        test_loss, correct = test(args, model, test_loader)
        writer2.add_scalar('test_loss_{}'.format(n_acqs), test_loss, epoch) # TODO nice to record all this, but will result in 200 scalar plots. I don't think I want to see them all later on
        writer2.add_scalar('test_correct_{}'.format(n_acqs), correct, epoch)
        max_correct = max(correct, max_correct)
    # record max_correct
    writer2.add_scalar('max_correct', max_correct, n_acqs)


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

    # pretrain_sampler = SubsetRandomSampler(pretrain_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)
    # pool_sampler = SubsetRandomSampler(pool_idx)

    # pretrain_loader = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.pretrain_batch_size, sampler=pretrain_sampler)
    # valid_loader = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.valid_batch_size, sampler=valid_sampler)
    # pool_loader = torch.utils.data.DataLoader(
        # train_data, batch_size=1, sampler=pool_sampler) # we'll draw batches of one example and on which to eval acq func
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.test_batch_size) # pytorch MNIST example shuffles test set, but seems unnecessary
    
    # pretraining
    # repeat for various choices of lambda:
    #    initial training on 20 points (random but *balanced*)
    #    compute validation error on 100 points
    # select lamba/model with lowest validation error
    # weight_decay = compute_weight_decay(pretrain_loader, valid_loader, args, writers)
    weight_decay = 1.0 # for now, let's not bother optimizing weight decay
    n_acqusitions = 0
    model = BayesianCNN()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    train_loader = get_train_loader(train_data, acq_idx, args)
    fit(model, optimizer, train_loader, test_loader, writers, args, n_acqusitions, epochs=50) # do pretraining on 20 examples (not quite clear if Gal does this here, but I think so)
    
    while n_acqusitions < 20: # TODO change to 1000
        # acquire 10 points from train_data according to acq_func
        new_idx = args.acq_func(train_data, pool_idx, model)
        acq_idx.update(new_idx)
        pool_idx.difference_update(new_idx)
        n_acqusitions += 10
        # reinitalise model
        model = BayesianCNN()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        # train model to convergence on all points acquired so far, computing test error as we go
        # train_loader built from train_data and acq_idx TODO: also includes pretrain_loader examples?? if not, it's a little weird because they are implicitly used in round 0 in computing acquisition function...
        train_loader = get_train_loader(train_data, acq_idx, args)
        fit(model, optimizer, train_loader, test_loader, writers, args, n_acqusitions, epochs=50)

    import ipdb; ipdb.set_trace()


def acq_random(train_data, pool_idx, model, n_acqs=10):
    """
    Random acquisition baseline.
    """
    new_idx = np.random.choice(pool_idx, size=n_acqs, replace=False)
    return set(new_idx)

def acq_BALD(train_data, pool_idx, model, n_acqs=10):
    for i in pool_idx:
        point = train_data[i]
        # apply BALD on point and record info gain
    # return 10 idx that maximise BALD


if __name__ == '__main__':
    main()