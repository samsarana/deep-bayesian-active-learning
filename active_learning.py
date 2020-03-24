import random

def acq_random(train_data, pool_idx, model, args):
    """
    Random acquisition baseline.
    """
    new_idx = random.sample(pool_idx, k=args.acqs_per_round) # random sample without replacement
    return set(new_idx)

def acq_BALD(train_data, pool_idx, model, args):
    for i in pool_idx:
        point = train_data[i]
        # apply BALD on point and record info gain
    # return 10 idx that maximise BALD

def acq_max_ent():
    pass

def acq_mean_std():
    pass

def acq_var_ratios():
    pass