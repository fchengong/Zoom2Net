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

def load_model(config, model_path):
    model = TSTransformerEncoder(
        window_size=config.window_size,
        feat_dim=config.feat_dim,
        d_model=config.d_model,
        n_heads=config.n_heads,
        num_layers=config.num_layers,
        max_len=1,
        dim_feedforward=config.dim_feedforward,
        dim_output=1,
        dropout=0.2,
        activation='relu',
        norm='LayerNorm'
        )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def impute_data(config, model, X_test, Y_test, maximum_deleted, minimum_deleted, y1_min, y1_max, y2_min, y2_max):
    res_true = []
    res_pred = []
    fiat_max_index = -8
    biat_max_index = -7
    fwd_bytes_max_index = -2
    bwd_bytes_max_index = -1
    
    with silence_stdout():
        for i in range(len(X_test)):
            fiat_max = (X_test[i][fiat_max_index]*(maximum_deleted[fiat_max_index]-minimum_deleted[fiat_max_index]) +\
                            minimum_deleted[fiat_max_index] - y2_min)
            biat_max = (X_test[i][biat_max_index]*(maximum_deleted[biat_max_index]-minimum_deleted[biat_max_index]) +\
                            minimum_deleted[biat_max_index] - y2_min)
            num_pkt_fwd = int(X_test[i][1]*(maximum_deleted[1]-minimum_deleted[1]) + minimum_deleted[1])
            num_pkt_bwd = int(X_test[i][2]*(maximum_deleted[2]-minimum_deleted[2]) + minimum_deleted[2])
            fwd_bytes_max = (X_test[i][fwd_bytes_max_index]*(maximum_deleted[fwd_bytes_max_index]-minimum_deleted[fwd_bytes_max_index]) +\
                            minimum_deleted[fwd_bytes_max_index] - y1_min)
            bwd_bytes_max = (X_test[i][bwd_bytes_max_index]*(maximum_deleted[bwd_bytes_max_index]-minimum_deleted[bwd_bytes_max_index]) +\
                            minimum_deleted[bwd_bytes_max_index] - y1_min)
                            
            x = inference(model, X_test[i])[0][0].cpu().numpy()
            # x = x[0].reshape((4,40))
            x[0:40] *= (y1_max - y1_min)
            x[40:80] *= (y2_max - y2_min)
            x[80:120] *= (y1_max - y1_min)
            x[120:160] *= (y2_max - y2_min)
            x = x.reshape((4,40))
            res = linear_programming(fiat_max, biat_max, fwd_bytes_max, bwd_bytes_max, num_pkt_fwd, num_pkt_bwd, x)
            if res == None:
                continue
            # print(f"res: {len(res)}, {res}")
            # res_pred.append(np.concatenate((res[0], res[1], res[2], res[3])))
            res_pred.append(np.array(res))
            res_true.append(np.concatenate((Y_test[i][0], Y_test[i][1], Y_test[i][2], Y_test[i][3])))
    return res_pred, res_true


# Initiate ILP model and solve
def linear_programming(fiat_max, biat_max, fwd_bytes_max, bwd_bytes_max, num_pkt_fwd, num_pkt_bwd, ml_output):
    model = gp.Model(name="model")
    model.Params.LogToConsole = 0

    x = model.addMVar(shape=(4,40), vtype=GRB.INTEGER, name="x")
    for i in range(num_pkt_fwd, 40):
        model.addConstr(x[0][i] == 0)
        model.addConstr(x[1][i] == 0)
    for i in range(num_pkt_bwd, 40):
        model.addConstr(x[2][i] == 0)
        model.addConstr(x[3][i] == 0)
        
    model.addConstr(x[1][0] == 0)
    model.addConstr(x[3][0] == 0)

    # print(fiat_min, biat_min)
    # fiat max
    maxi_fiat = model.addVar(name="maxi_fiat")
    model.addConstr(maxi_fiat == fiat_max)
    model.addConstr(maxi_fiat == gp.max_(x[1][i] for i in range(num_pkt_fwd)))
    # biat max
    maxi_biat = model.addVar(name="maxi_biat")
    model.addConstr(maxi_biat == biat_max)
    model.addConstr(maxi_biat == gp.max_(x[3][i] for i in range(num_pkt_bwd)))
    # fpkt max
    maxi_fpkt = model.addVar(name="maxi_fpkt")
    model.addConstr(maxi_fpkt == fwd_bytes_max)
    model.addConstr(maxi_fpkt == gp.max_(x[0][i] for i in range(num_pkt_fwd)))
    # bpkt max
    maxi_bpkt = model.addVar(name="maxi_bpkt")
    model.addConstr(maxi_bpkt == bwd_bytes_max)
    model.addConstr(maxi_bpkt == gp.max_(x[2][i] for i in range(num_pkt_bwd)))

    t1 = model.addMVar((4,40), name="t1")   # auxiliary variables
    t2 = model.addMVar((4,40), name="t2")   # auxiliary variables

    for i in range(4):
        for j in range(40):
            model.addConstr(t1[i][j] - t2[i][j] == ml_output[i][j] - x[i][j])
    
    linexpr = gp.quicksum((t1[i][j] + t2[i][j]) for i in range(4) for j in range(40))
    model.setObjective(linexpr)
    model.optimize()
    if model.status == 3:
        print("None")
        return None
    else:
        r = []
        for v in model.getVars():
            if "x[" in v.VarName:
                # print(v.VarName)
                r.append(v.X)
        return r

def z2n_new_features(model, input, output, y1_min, y1_max, y2_min, y2_max, maximum_deleted, minimum_deleted, flat_res):
    fiat_total_z2n = []
    biat_total_z2n = []
    fiat_max_index = -8
    biat_max_index = -7
    fwd_bytes_max_index = -2
    bwd_bytes_max_index = -1
    with silence_stdout():
        for i in range(len(input)):
            x = inference(model, input[i])[0].cpu().numpy()

            fiat_max = (input[i][fiat_max_index]*(maximum_deleted[fiat_max_index]-minimum_deleted[fiat_max_index]) +\
                            minimum_deleted[fiat_max_index] - y2_min)
            biat_max = (input[i][biat_max_index]*(maximum_deleted[biat_max_index]-minimum_deleted[biat_max_index]) +\
                            minimum_deleted[biat_max_index] - y2_min)
            num_pkt_fwd = int(input[i][1]*(maximum_deleted[1]-minimum_deleted[1]) + minimum_deleted[1])
            num_pkt_bwd = int(input[i][2]*(maximum_deleted[2]-minimum_deleted[2]) + minimum_deleted[2])
            fwd_bytes_max = (input[i][fwd_bytes_max_index]*(maximum_deleted[fwd_bytes_max_index]-minimum_deleted[fwd_bytes_max_index]) +\
                            minimum_deleted[fwd_bytes_max_index] - y1_min)
            bwd_bytes_max = (input[i][bwd_bytes_max_index]*(maximum_deleted[bwd_bytes_max_index]-minimum_deleted[bwd_bytes_max_index]) +\
                            minimum_deleted[bwd_bytes_max_index] - y1_min)
            x = x[0].reshape((4,40))
            res = linear_programming(fiat_max, biat_max, fwd_bytes_max, bwd_bytes_max, num_pkt_fwd, num_pkt_bwd, x)
            res = np.array(res).reshape((4,40))
            # x = post_process_iat(input[i], flat_res[i][0][4], x, log=False)
            fiat_total_z2n.append(sum(res[1])* (y2_max - y2_min))
            biat_total_z2n.append(sum(res[3])* (y2_max - y2_min))
    fiat_mean_z2n = []
    biat_mean_z2n = []
    duration_z2n = []
    for i in range(len(fiat_total_z2n)):
        fiat_mean_z2n.append(fiat_total_z2n[i] / (flat_res[i][0][2]-1))
        biat_mean_z2n.append(biat_total_z2n[i] / (flat_res[i][0][3]-1))
        if fiat_total_z2n[i] > biat_total_z2n[i]:
            duration_z2n.append(fiat_total_z2n[i])
        else:
            duration_z2n.append(biat_total_z2n[i])
    fb_psec_z2n = []
    with silence_stdout():
        for i in range(len(input)):
            x = inference(model, input[i])[0].cpu().numpy()

            fiat_max = (input[i][fiat_max_index]*(maximum_deleted[fiat_max_index]-minimum_deleted[fiat_max_index]) +\
                            minimum_deleted[fiat_max_index] - y2_min)
            biat_max = (input[i][biat_max_index]*(maximum_deleted[biat_max_index]-minimum_deleted[biat_max_index]) +\
                            minimum_deleted[biat_max_index] - y2_min)
            num_pkt_fwd = int(input[i][1]*(maximum_deleted[1]-minimum_deleted[1]) + minimum_deleted[1])
            num_pkt_bwd = int(input[i][2]*(maximum_deleted[2]-minimum_deleted[2]) + minimum_deleted[2])
            fwd_bytes_max = (input[i][fwd_bytes_max_index]*(maximum_deleted[fwd_bytes_max_index]-minimum_deleted[fwd_bytes_max_index]) +\
                            minimum_deleted[fwd_bytes_max_index] - y1_min)
            bwd_bytes_max = (input[i][bwd_bytes_max_index]*(maximum_deleted[bwd_bytes_max_index]-minimum_deleted[bwd_bytes_max_index]) +\
                            minimum_deleted[bwd_bytes_max_index] - y1_min)
            x = x[0].reshape((4,40))
            res = linear_programming(fiat_max, biat_max, fwd_bytes_max, bwd_bytes_max, num_pkt_fwd, num_pkt_bwd, x)
            res = np.array(res).reshape((4,40))
            total = sum(np.concatenate((res[0], res[2]))) * (y1_max - y1_min)
            duration = duration_z2n[i] 
            fb_psec_z2n.append(total / (duration / (10**6)))
        
    return fiat_total_z2n, biat_total_z2n, fiat_mean_z2n, biat_mean_z2n, duration_z2n, fb_psec_z2n