#!/bin/bash
python main.py --random_acq
python main.py --acq_func_ID info_gain
python main.py --acq_func_ID max_ent
python main.py --acq_func_ID mean_std
python main.py --acq_func_ID var_ratio