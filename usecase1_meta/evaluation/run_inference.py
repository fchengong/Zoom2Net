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

def load_model(config, model_path, d_model, n_heads, dim_feedforward, zoom_in_factor, window_size):
    model = TSTransformerEncoder(
        window_size=window_size,
        feat_dim=config.feat_dim,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=config.num_layers,
        max_len=4*int(window_size / zoom_in_factor),
        dim_feedforward=dim_feedforward,
        dim_output=1,
        dropout=0.2,
        activation='relu',
        norm='LayerNorm'
        )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def impute_data(config, model, test_dataset, rackdata_len, ingressBytes_max, timing, WINDOW_SIZE,\
                WINDOW_SKIP, COARSE):
    indexes = []
    for i in range(92):
        a = list(range(i))
        b = list(range(i+1,92))
        indexes.append((a+b))
    num_intervals = len(np.arange(0,2000,WINDOW_SKIP))-1
    num_WINDOW = len(np.arange(0,2000,WINDOW_SIZE))
    skipped = WINDOW_SIZE // WINDOW_SKIP
    res_true = np.zeros((rackdata_len, 92, num_WINDOW, WINDOW_SIZE))
    res_pred = np.zeros((rackdata_len, 92, num_WINDOW, WINDOW_SIZE))
    time_spend = []
    with silence_stdout():
        for server in range(92):
            for i in range(rackdata_len):
                cnt = 0
                for j in range(i*num_intervals, (i+1)*num_intervals):
                    if (j < num_intervals and j % skipped == 0) or \
                    (j >= num_intervals * i and (j - num_intervals * i) % skipped == 0):
                        t = test_dataset[j][0][:,[0,2,4,5],:]
                        t[:,0,:] = t[:,0,:]/3
                        t[:,3,:] = t[:,3,:]/6
                        b = (np.sum(t[indexes[server]], axis=0))
                        a = np.expand_dims(t[server], axis=0)
                        b = np.expand_dims(b, axis=0)/91/2
                        d = np.concatenate((a,b))
                        if timing:
                            start_time = time.time()
                        x = inference(model, d, WINDOW_SIZE=WINDOW_SIZE, COARSE=COARSE)[0][0].cpu().numpy()
                        summation = [sum(test_dataset[j][1][server][i*COARSE:(i+1)*COARSE]*ingressBytes_max) \
                                        for i in range(WINDOW_SIZE // COARSE)]    
                        fixed = test_ml_prediction_parallel(summation, x*ingressBytes_max, \
                                        test_dataset[j][0][server][4], config, ingressBytes_max)
                        if timing:
                            end_time = time.time()
                            execution_time = end_time - start_time
                            time_spend.append(execution_time/(WINDOW_SIZE // COARSE))
                        res_true[i,server,cnt,:] = (test_dataset[j][1][server])
                        res_pred[i,server,cnt,:] = fixed
                        cnt += 1
                    
    res_true = np.reshape(res_true, (rackdata_len,92,num_WINDOW*WINDOW_SIZE))
    res_pred = np.reshape(res_pred, (rackdata_len,92,num_WINDOW*WINDOW_SIZE))
    res_true = np.reshape(res_true, (rackdata_len*92,num_WINDOW*WINDOW_SIZE))
    res_pred = np.reshape(res_pred, (rackdata_len*92,num_WINDOW*WINDOW_SIZE))

    return res_pred, res_true, time_spend

# Use ILP in parallel
def parallel(i, summation, output, congestion, ingressBytes_max, COARSE):
    ml_output = output[i*COARSE:(i+1)*COARSE]
    incongestion = congestion[i]
    sum_interval = round(summation[i])

    r = linear_programming(sum_interval, ml_output, incongestion, ingressBytes_max, COARSE)
    if r is None:
        print("None")
        return None
    else:
        return [i, r]

def is_capsule(o):
    t = type(o)
    return t.__module__ == 'builtins' and t.__name__ == 'PyCapsule'

def test_ml_prediction_parallel(summation, output, congestion, config, ingressBytes_max): 
    WINDOW_SIZE = config.window_size
    COARSE = config.zoom_in_factor
    result = np.zeros(WINDOW_SIZE)
    pool = Pool(WINDOW_SIZE//COARSE)
    
    for return_val in pool.map(partial(parallel, summation=(summation), output=output, congestion=congestion, \
                                ingressBytes_max=ingressBytes_max, COARSE=COARSE), np.arange(WINDOW_SIZE//COARSE)):
        if return_val != None:
            result[return_val[0]*COARSE:(return_val[0]+1)*COARSE] = return_val[1]
    
    return result

# Initiate ILP model and solve
def linear_programming(summation, ml_output, incongestion, ingressBytes_max, COARSE):
    model = gp.Model(name="model")
    model.Params.LogToConsole = 0

    x = model.addMVar(shape=COARSE, vtype=GRB.INTEGER, name="x")
    model.addConstr(x.sum() == summation, name="c")

    a = model.addVars(COARSE, name="a")  
    b = model.addVars(COARSE, name="b")  
    t1 = model.addVars(COARSE, name="t1") 
    t2 = model.addVars(COARSE, name="t2")  

    for i in range(COARSE):
        model.addConstr(t1[i] - t2[i] == ml_output[i] - x[i])
        model.addConstr(b[i] == t1[i] + t2[i])
    if incongestion > 0:
        maxi = model.addVar(name="maxi")
        model.addConstr(maxi == gp.max_(x[i] for i in range(COARSE)))
        model.addConstr(maxi >= ingressBytes_max/2)
        
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
