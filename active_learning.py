import random, torch, time, logging
import numpy as np
from utils import make_dataloader

def make_acquisitions(train_data, pool_idx, model, args):
    """
    Chooses pool points that maximise the acquisition function given in args.acq_func
    (Or acquires points uniformly at random if args.random_acq == True.)
    Returns indices of the top `args.acqs_per_round` points from the pool.
    The elements of train_data at pool_idx are the "pool points".

    Parameters
    ----------
    train_data: torch.utils.data.Dataset
        PyTorch dataset, superset of the pool data
    pool_idx: set
        indices specifying which points in train_data are in the pool
    model: torch.nn.Module
        find information gained about this PyTorch model
    args: Namespace object
        experiment arguments from argparse, including acq_func

    Returns
    -------
    new_idx: set
        indices of pool points which maximise acquisition function
        len(new_idx) = args.acqs_per_round
    mean_info: float (or None)
        the mean "informativenss" of these points, measuring using the given
        acqusition function (entropy, variation ratio or standard deviation)
    """
    if args.random_acq:
        new_idx = set(random.sample(pool_idx, k=args.acqs_per_round)) # random sample without replacement
        mean_info = None
    else:
        best_ent_idx = np.zeros(shape=(0,2), dtype=np.float64) # array for storing top 10 (entropy, idx) pairs
        start = time.time()
        pool_loader = make_dataloader(train_data, pool_idx, args.test_batch_size, random=False) # note 1
        with torch.no_grad():
            for data, _, idx in pool_loader:
                logprobs = model.forward_stochastic(data, k=args.dropout_samples).double() # do entropy calcs in double precision
                # model outputs logprobs (final layer is log_softmax(.))
                # this is for numerical stability in softmax computation
                # convert these back to probs to do entropy calculations
                probs = logprobs.exp()
                info = args.acq_func(probs)
                # add new (entropy, index) tuples to array of best so far
                new_ent_idx = np.column_stack((info, idx))
                all_ent_idx = np.concatenate((new_ent_idx, best_ent_idx), axis=0)
                # sort by entropy and take top 10 so far
                sorted_ind = all_ent_idx[:,0].argsort()
                best_ent_idx = all_ent_idx[sorted_ind][-args.acqs_per_round:]
                logging.info('Computing entropy for points with (original) indices {}-{} in pool'.format(
                    idx[0], idx[-1]))
        
        assert best_ent_idx.shape == (args.acqs_per_round, 2)
        end = time.time()
        logging.info("Time taken for {} acquisitions: {}".format(args.acqs_per_round, end - start))
        new_idx = set(best_ent_idx[:,1].astype(int))
        mean_info = best_ent_idx[:,0].mean()
    return new_idx, mean_info

def info_gain(probs):
    # mean over dropout samples
    p_yc = probs.mean(dim=-1)
    # compute entropy and sum over class dimension (giving total uncertainty)
    H_y = - (p_yc * p_yc.log()).sum(dim=-1).numpy()
    # compute aleatoric uncertainty
    E_H_y = -(probs * probs.log()).sum(dim=1).mean(dim=-1).numpy()
    # deduce epistemic uncertainty
    info = H_y - E_H_y
    return info

def max_ent(probs):
    # mean over dropout samples
    p_yc = probs.mean(dim=-1)
    # compute entropy and sum over class dimension
    H_y = - (p_yc * p_yc.log()).sum(dim=-1).numpy()
    return H_y

def mean_std(probs):
    # standard deviation over dropout samples
    # math in Gal's paper shows him using population std (but I'm a little unsure)
    stds = probs.std(dim=-1, unbiased=False)
    # take mean of stds over classes
    mean_stds = stds.mean(dim=-1).numpy()
    return mean_stds

def var_ratio(probs):
    # mean over dropout samples
    p_yc = probs.mean(dim=-1)
    # var_ratios = (1 - max prob over classes)
    var_ratios = 1 - p_yc.max(dim=-1).values.numpy()
    return var_ratios