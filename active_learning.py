import random

def acq_random(train_data, pool_idx, model, n_acqs=10):
    """
    Random acquisition baseline.
    """
    new_idx = random.sample(pool_idx, k=n_acqs) # random sample without replacement
    return set(new_idx)

def acq_BALD(train_data, pool_idx, model, n_acqs=10):
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