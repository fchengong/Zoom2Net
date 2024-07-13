import gurobipy as gp
from gurobipy import GRB
from multiprocessing import Pool
from functools import partial
import numpy as np
from itertools import repeat
import time
import os
import sys
from contextlib import contextmanager
import torch
import itertools

from model_training.utils import inference
from model_training.transformer import TSTransformerEncoder

@contextmanager
def silence_stdout():
    old_target = sys.stdout
    try:
        with open(os.devnull, "w") as new_target:
            sys.stdout = new_target
            yield new_target
    finally:
        sys.stdout = old_target

def load_model(config, model_path, d_model, n_heads, dim_feedforward):
    model = TSTransformerEncoder(
        window_size=config.window_size,
        feat_dim=config.feat_dim,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=config.num_layers,
        max_len=config.window_size,
        dim_feedforward=dim_feedforward,
        dim_output=1,
        dropout=0.2,
        activation='relu',
        norm='LayerNorm'
        )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def impute_data(config, model, test_dataset, timing, maxsentBytes, data_3s_train, \
                data_3s_test, data_3to6s_train, data_3to6s_test, data_6to9s_train, data_6to9s_test, \
                index_of_3s, index_of_3to6s, index_of_6to9s):
    pred9s_z2n = []
    true9s_z2n = []
    pred6s_z2n = []
    true6s_z2n = []
    pred3s_z2n = []
    true3s_z2n = []
    with silence_stdout():
        for i in range(data_3s_test):
            s3 = inference(model, (test_dataset[i][0])).cpu().numpy()[0]
            s3 = fix(test_dataset, i, s3, config.window_size, maxsentBytes)
            ground_thruth_3s = test_dataset[i][1]
            a = index_of_3s[data_3s_train+i]
            b = np.where(index_of_3to6s == a)[0]
            c = np.where(index_of_6to9s == a)[0]
            exist_3to6 = False
            exist_6to9 = False
            if len(b) != 0 and b[0] > data_3to6s_train:
                exist_3to6 = True
                test_index = b[0] - data_3to6s_train + data_3s_test
                s3to6 = inference(model, (test_dataset[test_index][0])).cpu().numpy()[0]
                s3to6 = fix(test_dataset, test_index, s3to6, config.window_size, maxsentBytes)
                ground_thruth_3to6s = test_dataset[test_index][1]
            if exist_3to6 == True and len(c) != 0 and c[0] > data_6to9s_train:
                exist_6to9 = True
                test_index2 = c[0] - data_6to9s_train + data_3s_test + data_3to6s_test
                s6to9 = inference(model, (test_dataset[test_index2][0])).cpu().numpy()[0]
                s6to9 = fix(test_dataset, test_index2, s6to9, config.window_size, maxsentBytes)
                ground_thruth_6to9s = test_dataset[test_index2][1]
            if exist_3to6 == True and exist_6to9 == True:
                pred9s_z2n.append(np.concatenate((s3, s3to6, s6to9)))
                true9s_z2n.append(np.concatenate((ground_thruth_3s, ground_thruth_3to6s, ground_thruth_6to9s)))
            elif exist_3to6 == True and exist_6to9 == False:
                pred6s_z2n.append(np.concatenate((s3, s3to6)))
                true6s_z2n.append(np.concatenate((ground_thruth_3s, ground_thruth_3to6s)))
            elif exist_3to6 == False and exist_6to9 == False:
                pred3s_z2n.append(s3)
                true3s_z2n.append(ground_thruth_3s)
    pred9s_z2n = np.array(pred9s_z2n)
    true9s_z2n = np.array(true9s_z2n)
    pred6s_z2n = np.array(pred6s_z2n)
    true6s_z2n = np.array(true6s_z2n)
    pred3s_z2n = np.array(pred3s_z2n)
    true3s_z2n = np.array(true3s_z2n)

    return [true9s_z2n, true6s_z2n, true3s_z2n], [pred9s_z2n, pred6s_z2n, pred3s_z2n]

# Use ILP in parallel
def parallel(i, summation, maximum, ml): 
    prev = 0
    ml_output = ml[i]
    
    r = linear_programming(round(summation[i]), round(maximum[i]), ml_output)
    if r is None:
            print("None")
            return None
    else:
        return (i, r)

def is_capsule(o):
    t = type(o)
    return t.__module__ == 'builtins' and t.__name__ == 'PyCapsule'

def fix(test_dataset, index, s3, WINDOW_SIZE, maxsentBytes):
    result = np.zeros(WINDOW_SIZE)
    time_data = test_dataset[index][2]
    prev = 0
    summation = []
    maximum = []
    ml = []
    pairs = []
    cnt = 0
    for j in range(len(time_data)):
        t = int(time_data[j])
        if t == 0:
            break
        summation.append(sum(test_dataset[index][1][prev:t]*maxsentBytes))
        maximum.append(max(test_dataset[index][1][prev:t]*maxsentBytes))
        ml.append(s3[prev:t]*maxsentBytes)
        pairs.append((prev, t))
        prev = t
        cnt += 1

    pool = Pool(cnt)
    for return_val in (pool.imap_unordered(partial(parallel, summation=summation, maximum=maximum, ml=ml), \
                                np.arange(cnt))):
        if return_val != None:
            ind = return_val[0]
            result[pairs[ind][0] : pairs[ind][1]] = return_val[1]
    return result / maxsentBytes

# Initiate ILP model and solve
def linear_programming(summation, maximum, ml_output):
    model = gp.Model(name="model")
    model.Params.LogToConsole = 0
    length = len(ml_output)

    x = model.addMVar(shape=length, vtype=GRB.INTEGER, name="x")
    model.addConstr(x.sum() == summation, name="c")
    maxi = model.addVar(name="maxi")
    model.addConstr(maxi == maximum)
    model.addConstr(maxi == gp.max_(x[i] for i in range(length)))

    a = model.addVars(length, name="a")   # auxiliary variables
    b = model.addVars(length, name="b")   # auxiliary variables
    t1 = model.addVars(length, name="t1")   # auxiliary variables
    t2 = model.addVars(length, name="t2")   # auxiliary variables

    for i in range(length):
        model.addConstr(t1[i] - t2[i] == ml_output[i] - x[i])
    for i in range(length):
         model.addConstr(b[i] == t1[i] + t2[i])
    m = model.addVar(name="m")
    model.addConstr(m == gp.max_(b))
    model.setObjective(m)
    model.optimize()
    if model.status == 3:
        print("None")
        return None
    else:
        r = []
        for v in model.getVars():
            if "x[" in v.VarName:
                r.append(v.X)
        return r
