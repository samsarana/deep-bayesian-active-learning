import random, logging, argparse, torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models import BayesianCNN
from sampling import balanced_sample
from utils import *

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
    logging.info('Begin train/test for {} acquisitions'.format(cumulative_acqs))
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