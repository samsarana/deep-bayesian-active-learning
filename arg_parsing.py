import argparse
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
    parser.add_argument('--test_batch_size', type=int, default=1000, help='Batch size for testing on 10k examples. This should be as large as possible without crashing the C/GPU')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--test_interval', type=int, default=10, help='Number of epochs to train before testing')
    # active learning settings
    parser.add_argument('--dropout_samples', type=int, default=20, help='Number of MC-dropout samples from approximate posterior') # NB Gal uses 100?
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