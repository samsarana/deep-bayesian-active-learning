import random, torch, time, logging
import numpy as np
from utils import make_dataloader

def acq_random(train_data, pool_idx, model, args):
    """
    Random acquisition baseline.
    """
    new_idx = random.sample(pool_idx, k=args.acqs_per_round) # random sample without replacement
    return set(new_idx), None

def acq_BALD(train_data, pool_idx, model, args):
    pass

def acq_max_ent(train_data, pool_idx, model, args):
    """
    Chooses pool points that maximise the predictive entopy (Shannon, 1948).
    Returns indices of the top `args.acqs_per_round` points from the pool.
    The elements of train_data at pool_idx are the "pool points".

    Parameters
    ----------
    train_data: torch.utils.data.Dataset
        PyTorch dataset, superset of the pool data
    pool_idx: set
        indices specifying which points in train_data are in the pool
    model: torch.nn.Module
        find prective entropy of points wrt this PyTorch model
    args: Namespace object
        experiment arguments from argparse

    Returns
    -------
    new_idx: set
        indices of pool points which maximise predictive entropy
        len(new_idx) = args.acqs_per_round
    mean_entropy: float
        the mean entropy of those selected pool points
    """
    best_ent_idx = np.zeros(shape=(0,2), dtype=np.float64) # array for storing top 10 (entropy, idx) pairs
    start = time.time()
    pool_loader = make_dataloader(train_data, pool_idx, args.test_batch_size, random=False) # note 1
    with torch.no_grad():
        for data, _, idx in pool_loader:
            output = model.forward_stochastic(data, k=args.dropout_samples).double() # do entropy calcs in double precision
            # mean over dropout samples
            log_p_yc_xD = output.mean(dim=-1)
            # compute entropy and sum over class dimension TODO write some checks to verify entropy calculation
            H_y_xD = - (log_p_yc_xD * log_p_yc_xD.exp()).sum(dim=-1).numpy()
            # add new (entropy, index) tuples to array of best so far
            import ipdb;ipdb.set_trace()
            new_ent_idx = np.column_stack((H_y_xD, idx))
            all_ent_idx = np.concatenate((new_ent_idx, best_ent_idx), axis=0)
            # sort by entropy and take top 10 so far
            sorted_ind = all_ent_idx[:,0].argsort()
            best_ent_idx = all_ent_idx[sorted_ind][-args.acqs_per_round:]
            logging.info('Computing entropy for points with (original) indices {}-{} in pool'.format(
                idx[0], idx[-1]))
    
    assert best_ent_idx.shape == (args.acqs_per_round, 2)
    end = time.time()
    logging.info("Time taken for {} acquisitions: {}".format(args.acqs_per_round, end - start))
    mean_entropy = best_ent_idx[:,0].mean()
    new_idx = set(best_ent_idx[:,1].astype(int))
    return new_idx, mean_entropy

def acq_mean_std():
    pass

def acq_var_ratios():
    pass