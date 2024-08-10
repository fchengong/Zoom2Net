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

from model_training.utils import inference, inference_withoutCCA
from model_training.transformer import TSTransformerEncoder
from preprocessing.preprocessor3 import convert_even, convert_odd

@contextmanager
def silence_stdout():
    old_target = sys.stdout
    try:
        with open(os.devnull, "w") as new_target:
            sys.stdout = new_target
            yield new_target
    finally:
        sys.stdout = old_target

def load_model(config, model_path, d_model, n_heads, dim_feedforward, max_len, zoom_in_factor, window_size):
    model = TSTransformerEncoder(
        window_size=window_size,
        feat_dim=config.feat_dim,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=config.num_layers,
        max_len=max_len,
        dim_feedforward=dim_feedforward,
        dim_output=1,
        dropout=0.2,
        activation='relu',
        norm='LayerNorm'
        )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def impute_data(config, even_model, odd_model, test_dataset, rackdata_len, res_queue_max, timing, WINDOW_SIZE,\
                WINDOW_SKIP, COARSE):

    even = np.arange(0,8)
    indexes = []
    for i in range(8):
        a = list(range(i))
        b = list(range(i+1,8))
        indexes.append((a+b))
    res_true = np.zeros((rackdata_len, 16, 33, 300))
    res_pred = np.zeros((rackdata_len, 16, 33, 300))
    num_intervals = 33
    skipped = WINDOW_SIZE // WINDOW_SKIP
    time_spend = []
    with silence_stdout():
        for q in even:
            feature_ports = q
            label_ports_even = q*2
            label_ports_odd = q*2+1
            for i in range(rackdata_len):
                cnt = 0
                for j in range(i*num_intervals, (i+1)*num_intervals):
                    if (j < num_intervals and j % skipped == 0) or \
                    (j >= num_intervals * i and (j - num_intervals * i) % skipped == 0):
                        converted = convert_even(test_dataset[j][0])
                        b = (np.sum(converted[indexes[feature_ports]], axis=0))
                        a = np.expand_dims(converted[feature_ports], axis=0)
                        b = np.expand_dims(b, axis=0)/7
                        if timing:
                            start_time = time.time()
                        x = inference_withoutCCA(even_model, np.concatenate((a,b)), COARSE=COARSE)[0].cpu().numpy()
                        maximum = [max(test_dataset[j][1][label_ports_even][i*COARSE:(i+1)*COARSE]\
                                                            *res_queue_max[label_ports_even]) \
                                            for i in range(WINDOW_SIZE // COARSE)]   
                        periodic = [(test_dataset[j][1][label_ports_even][i*COARSE]\
                                                            *res_queue_max[label_ports_even]) \
                                            for i in range(WINDOW_SIZE // COARSE)]  
                        fixed = test_ml_prediction_parallel(maximum, periodic, \
                                                            x*res_queue_max[label_ports_even], config)
                        if timing:
                                end_time = time.time()
                                execution_time = end_time - start_time
                                time_spend.append(execution_time/(WINDOW_SIZE // COARSE))
                        res_true[i,label_ports_even,cnt,:] = test_dataset[j][1][label_ports_even]
                        res_pred[i,label_ports_even,cnt,:] = fixed/res_queue_max[label_ports_even]
                        
                        converted = convert_odd(test_dataset[j][0])
                        b = (np.sum(converted[indexes[feature_ports]], axis=0))
                        if j < len(test_dataset)//2:
                            c = np.zeros((1,6))
                        else:
                            c = np.ones((1,6))
                        a = np.expand_dims(np.concatenate((converted[feature_ports],c)), axis=0)
                        b = np.expand_dims(np.concatenate((b,c)), axis=0)/7
                        if timing:
                            start_time = time.time()
                        x = inference(odd_model, np.concatenate((a,b)), COARSE=COARSE)[0].cpu().numpy()
                        maximum = [max(test_dataset[j][1][label_ports_odd][i*COARSE:(i+1)*COARSE]\
                                                            *res_queue_max[label_ports_odd]) \
                                            for i in range(WINDOW_SIZE // COARSE)] 
                        periodic = [(test_dataset[j][1][label_ports_odd][i*COARSE]\
                                                            *res_queue_max[label_ports_odd]) \
                                            for i in range(WINDOW_SIZE // COARSE)]   
                        fixed = test_ml_prediction_parallel(maximum, periodic, \
                                                            x*res_queue_max[label_ports_odd], config)
                        # fixed = x
                        if timing:
                                end_time = time.time()
                                execution_time = end_time - start_time
                                time_spend.append(execution_time/(WINDOW_SIZE // COARSE))
                        res_true[i,label_ports_odd,cnt,:] = test_dataset[j][1][label_ports_odd]
                        res_pred[i,label_ports_odd,cnt,:] = fixed/res_queue_max[label_ports_odd]
                        cnt += 1
    res_true = np.reshape(res_true, (rackdata_len,16,9900))
    res_pred = np.reshape(res_pred, (rackdata_len,16,9900))

    return res_pred, res_true, time_spend

# Use ILP in parallel
def parallel(i, maximum, priodic, output, COARSE):
    ml_output = output[i*COARSE:(i+1)*COARSE]
    max_interval = round(maximum[i])
    priodic_interval = round(priodic[i])

    r = linear_programming(max_interval, priodic_interval, ml_output, COARSE)
    if r is None:
        print("None")
        return None
    else:
        return [i, r]

def is_capsule(o):
    t = type(o)
    return t.__module__ == 'builtins' and t.__name__ == 'PyCapsule'

def test_ml_prediction_parallel(maximum, priodic, output, config): 
    WINDOW_SIZE = config.window_size
    COARSE = config.zoom_in_factor
    result = np.zeros(WINDOW_SIZE)
    pool = Pool(WINDOW_SIZE//COARSE)
    
    for return_val in pool.map(partial(parallel, maximum=maximum, priodic=priodic, output=output, COARSE=COARSE),\
                                 np.arange(WINDOW_SIZE//COARSE)):
        if return_val != None:
            result[return_val[0]*COARSE:(return_val[0]+1)*COARSE] = return_val[1]
    
    return result

# Initiate ILP model and solve
def linear_programming(maximum, periodic, ml_output, COARSE):
    model = gp.Model(name="model")
    model.Params.LogToConsole = 0

    x = model.addMVar(shape=COARSE, vtype=GRB.INTEGER, name="x")
    maxi = model.addVar(name="maxi")
    model.addConstr(maxi == maximum)
    model.addConstr(maxi == gp.max_(x[i] for i in range(COARSE)))

    period = model.addVar(name="period")
    model.addConstr(period == periodic)
    model.addConstr(period == x[0])

    a = model.addVars(COARSE, name="a")   # auxiliary variables
    t1 = model.addVars(COARSE, name="t1")   # auxiliary variables
    t2 = model.addVars(COARSE, name="t2")   # auxiliary variables

    for i in range(COARSE):
        model.addConstr(t1[i] - t2[i] == ml_output[i] - x[i])
    
    linexpr = gp.quicksum((t1[i] + t2[i]) for i in range(COARSE))
    model.setObjective(linexpr)
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
