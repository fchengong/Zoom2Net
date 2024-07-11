from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool
import torch

from model_training.utils import inference
from model_training.transformer import TSTransformerEncoder
from evaluation.run_inference import load_model

def myFunc(e):
  return e[1]

def metric(x,y):
    a = x - y
    # b = [j for j in a if j>0]
    b = [j if j>0 else 0 for j in a]
    return np.average(a**2) + np.max(b)

def impute_training_data(model, train_dataset_2, coarse):
    # Get imputed training dataset to identify data to be fixed
    imputed_train = []
    for i in range(len(train_dataset_2)):
        x = inference(model, (train_dataset_2[i][0]), COARSE = coarse)
        imputed_train.append(x[0][0].cpu().numpy())
    return imputed_train

def find_closest_neighbor(train_dataset_2):
    def data_proc(i):
        dist = []
        for j in range(len(train_dataset_2)):
            if j == i:
                continue
            d = np.linalg.norm(train_dataset_2[i][0] - train_dataset_2[j][0])
            dist.append([j, d])
        dist.sort(key=myFunc)
        return dist[0:len(train_dataset_2)]
        
    closest_neighbor = np.zeros((len(train_dataset_2),len(train_dataset_2)-1, 2))
    for i in range(len(train_dataset_2)):
        distance = data_proc(i)
        closest_neighbor[i] = distance
        
    return closest_neighbor

def data_refinement(train_dataset_2, config):
    model_plain = load_model(config, "checkpoints/plain_model.torch", d_model=config.d_model, n_heads=config.n_heads, dim_feedforward=config.dim_feedforward, 
                                zoom_in_factor=config.zoom_in_factor, window_size=config.window_size)
    model_plain.eval()
    imputed_train_plain = impute_training_data(model_plain, train_dataset_2, config.zoom_in_factor)
    closest_neighbor = find_closest_neighbor(train_dataset_2)
    num_window = int(config.window_size / config.zoom_in_factor)

    real_added2 = []
    added_lists2 = []
    cnt2 = 0
    train_dataset_refined = []
    a2 = []
    d = []

    for i in range(len(train_dataset_2)//2, len(train_dataset_2)):
        # print(i)
        close = [i]
        dist = closest_neighbor[i]
        if metric(train_dataset_2[i][1], imputed_train_plain[i]) < 0.01:
            continue
        if max(imputed_train_plain[i]) < 0.15:
            thre = 0.0005
        else:
            thre = 0.025
        for j in dist:
            if j[1] > 4:    
                break
            a = mean_absolute_error(imputed_train_plain[i], imputed_train_plain[int(j[0])])
            if a < thre and metric(train_dataset_2[int(j[0])][1], imputed_train_plain[int(j[0])]) > 0.01:
                close.append(int(j[0]))
        if len(close) > 1:
            stop = False
            # Data with different fine-grained and similar coarse-grained
            for n,j in enumerate(added_lists2):
                if len(set(close) & set(j)) > len(close) * 0.8:
                    close = list(set(close) - (set(close) & set(real_added2)))
                    added_lists2[n] = list(set(added_lists2[n] + close))
                    stop = True
                    real_added2.extend(close)
                    break
            if stop == False:
                if not bool(set(close) & set(real_added2)):
                    added_lists2.append(close)
                    real_added2.extend(close)
            cnt2 += 1
    for close in added_lists2:
        sets = np.zeros((15,1000))
        l = 15 if len(close) > 15 else len(close)
        # l = len(close)
        for k in range(l):
            sets[k] = train_dataset_2[close[k]][1]
        for k in close:
            train_dataset_refined.append((train_dataset_2[k][0], train_dataset_2[k][1], sets, l,\
                np.zeros(num_window), np.zeros(num_window)))
    # ######## Add data not in problematic clusters ###########
    for i in range(len(train_dataset_2)):
        if i not in real_added2:
            sets = np.zeros((15,1000))
            sets[0] = (train_dataset_2[i][1])
            train_dataset_refined.append((train_dataset_2[i][0], train_dataset_2[i][1], sets, 1,\
                np.zeros(num_window), np.zeros(num_window)))
            
    return train_dataset_refined
