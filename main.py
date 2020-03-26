import random, logging, torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models import BayesianCNN
from active_learning import make_acquisitions
from arg_parsing import parse_arguments
from utils import load_data, make_dataloader, balanced_sample

def train(args, model, train_loader, optimizer):
    model.train()
    train_set_size = len(train_loader) * train_loader.batch_size
    train_loss = 0
    correct = 0
    for data, target, _ in train_loader:
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
    # model.eval()
    model.train() # use dropout approximation at test time
    test_set_size = len(test_loader) * test_loader.batch_size
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, _ in test_loader:
            # output = model(data)
            output = model.forward_stochastic(data, k=args.dropout_samples).mean(dim=-1) # use dropout approximation at test time
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    ave_test_loss = test_loss / test_set_size
    proportion_correct = correct / test_set_size
    return ave_test_loss, proportion_correct

def fit(model, optimizer, train_loader, test_loader, args, writers, i_round):
    writer1, writer2 = writers
    cumulative_acqs = args.acqs_pretrain + i_round * args.acqs_per_round
    max_test_correct = 0
    logging.info('Begin train/test after {} acquisitions'.format(cumulative_acqs))
    for epoch in range(1, args.epochs + 1): # TODO may to increase epochs for training on e.g. 1000 points to ensure convergence
        logging.info('Training, epoch {}'.format(epoch))
        train_loss, train_correct = train(args, model, train_loader, optimizer)
        cumulative_epoch = epoch + i_round * args.epochs
        writer1.add_scalar('loss', train_loss, cumulative_epoch)
        writer1.add_scalar('correct', train_correct, cumulative_epoch)
        # test on test set
        if epoch % args.test_interval == 0:
            logging.info('TESTING, epoch {}'.format(epoch))
            test_loss, test_correct = test(args, model, test_loader)
            writer2.add_scalar('loss', test_loss, cumulative_epoch)
            writer2.add_scalar('correct', test_correct, cumulative_epoch)
            max_test_correct = max(test_correct, max_test_correct)
    # record max_test_correct
    writer2.add_scalar('max_test_correct', max_test_correct, cumulative_acqs)


def compute_weight_decay(pretrain_loader, valid_loader, args, writers, i_round):
    """
    "All models are trained ...with a validation set
    of 100 points on which we optimise the weight decay."
    """
    writer1, writer2 = writers
    valid_corrects = []
    valid_set_size = len(valid_loader) * valid_loader.batch_size
    weight_decays = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    for weight_decay in weight_decays:
        logging.info('Pretraining with weight decay {}'.format(weight_decay))
        max_val_correct = 0
        model = BayesianCNN()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay) # NB Gal doesn't mention which optimizer they use
        for epoch in range(1, args.epochs + 1):
            pretrain_loss, pretrain_correct = train(args, model, pretrain_loader, optimizer)
            cumulative_epoch = epoch + i_round * args.epochs
            writer1.add_scalar('pretrain_loss_{}'.format(weight_decay), pretrain_loss, cumulative_epoch)
            writer1.add_scalar('pretrain_correct_{}'.format(weight_decay), pretrain_correct, cumulative_epoch)
            # test on validation set
            valid_loss, valid_correct = test(args, model, valid_loader)
            writer2.add_scalar('pretrain_loss_{}'.format(weight_decay), valid_loss, cumulative_epoch)
            writer2.add_scalar('pretrain_correct_{}'.format(weight_decay), valid_correct, cumulative_epoch)
            max_val_correct = max(valid_correct, max_val_correct)
        # record max correct
        valid_corrects.append(max_val_correct)
    optimal_weight_decay = weight_decays[np.argmax(np.array(valid_corrects))]
    logging.info('Optimal weight decay = {} with accuracy {:.3f} on validation set'.format(
        optimal_weight_decay, max(valid_corrects)
    ))
    return optimal_weight_decay


def main():
    # experiment settings via command line
    args = parse_arguments()
    # setup logging
    experiment_name = 'random' if args.random_acq else args.acq_func_ID
    args.logdir = './logs/{}/{}'.format(experiment_name, args.seed)
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
    # get idx of pretrain and validation data from train data
    pool_idx = set(range(len(train_data)))
    pretrain_idx, pool_idx = balanced_sample(train_data, n_classes=10, k=2, 
        idx_possible=pool_idx)
    valid_idx, pool_idx = balanced_sample(train_data, n_classes=10, k=10, 
        idx_possible=pool_idx) # Gal doesn't mention if validation set is balanced
    acq_idx = pretrain_idx.copy() # first 20 acquisitions are the pretraining data
    
    assert len(pretrain_idx) == 20
    assert len(valid_idx) == 100
    assert len(pool_idx) == len(train_data) - 120

    # make dataloaders
    train_loader = make_dataloader(train_data, args.train_batch_size, idx=acq_idx, random=True)
    valid_loader = make_dataloader(train_data, args.valid_batch_size, idx=valid_idx)
    test_loader  = make_dataloader(test_data, args.test_batch_size) # pytorch MNIST example shuffles test set, but seems unnecessary
    
    # pretraining
    # repeat for various choices of lambda:
    #    initial training on 20 points (random but *balanced*)
    #    compute validation error on 100 points
    # select lamba/model with lowest validation error
    weight_decay = compute_weight_decay(train_loader, valid_loader, args, writers, i_round=0)
    # weight_decay = 1.0
    model = BayesianCNN()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    # if args.acqs_pretrain > 0:
    fit(model, optimizer, train_loader, test_loader, args, writers, i_round=0) # do pretraining on 20 examples (not quite clear if Gal does this here, but I think so)
    
    # REMOVE
    # pool_idx.difference_update(set(range(1000, 60000))) # srink size of pool to do some checks
    # END REMOVE
    for i_round in range(1, args.rounds + 1):
        # acquire 10 points from train_data according to acq_func
        new_idx, mean_info_gain = make_acquisitions(train_data, pool_idx, model, args)
        if not args.random_acq: writer1.add_scalar('mean_info_gain', mean_info_gain, i_round)
        acq_idx.update(new_idx)
        pool_idx.difference_update(new_idx)
        # build new train_loader using updated acq_idx
        train_loader = make_dataloader(train_data, args.train_batch_size, idx=acq_idx, random=True)
        # reoptimize weight decay given updated, larger training set. Unclear if Gal does this, but seems natural
        weight_decay = compute_weight_decay(train_loader, valid_loader, args, writers, i_round)
        # reinitalise model
        model = BayesianCNN()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        # train model to convergence on all points acquired so far, computing test error as we go
        oldw1 = list(model.parameters())[0][0][0][0][0]
        fit(model, optimizer, train_loader, test_loader, args, writers, i_round)
        neww1 = list(model.parameters())[0][0][0][0][0]
        assert oldw1 != neww1, "fit(.) didn't update model parameters"


if __name__ == '__main__':
    main()