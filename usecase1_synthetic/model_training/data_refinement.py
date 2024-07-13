from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool
import torch

from model_training.utils import inference, inference_withoutCCA
from model_training.transformer import TSTransformerEncoder
from evaluation.run_inference import load_model
from preprocessing.preprocessor3 import process_data

def metric(x,y):
    a = x - y
    # b = [j for j in a if j>0]
    b = [j if j>0 else 0 for j in a]
    return np.average(a**2) + np.max(b)

def impute_training_data(model, train_dataset_2, coarse, CCA):
    # Get imputed training dataset to identify data to be fixed
    imputed_train = []
    for i in range(len(train_dataset_2)):
        if CCA:
            x = inference(model, (train_dataset_2[i][0]), COARSE = coarse)
            imputed_train.append(x[0][0].cpu().numpy())
        else:
            x = inference_withoutCCA(model, (train_dataset_2[i][0]), COARSE = coarse)
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
def myFunc(e):
    return e[1]
def data_refinement(train_dataset_2, train_dataset, test_dataset, config):
    model_plain = load_model(config, "checkpoints/plain_model.torch", d_model=40, n_heads=config.n_heads, dim_feedforward=20, 
                                max_len=36, zoom_in_factor=config.zoom_in_factor, window_size=config.window_size)
    model_plain.eval()
    plain_train_dataset, plain_test_dataset = process_data(config, 
                                                train_dataset, test_dataset, include_cca=False)
    imputed_train = impute_training_data(model_plain, plain_train_dataset, config.zoom_in_factor, CCA=False)
    closest_neighbor = find_closest_neighbor(plain_train_dataset)
    num_window = int(config.window_size / config.zoom_in_factor)
    num_period_sample = len(np.arange(0, config.window_size, int(config.zoom_in_factor/2)))
    added2 = []
    real_added2 = []
    cnt2 = 0
    train_dataset_refined = []
    a2 = []
    d = []
    added_lists2 = []

    for i in range(len(train_dataset_2)):
        close = [i]
        dist = closest_neighbor[i]
        if metric(train_dataset_2[i][1], imputed_train[i]) < 0.009:
            continue
        thre = 0.008 # 0.0005
        for j in dist:
            if j[1] > 2:       # 4.327383845978116:
                break
            a = mean_squared_error([imputed_train[i]], [imputed_train[int(j[0])]])
            if a < thre and metric(train_dataset_2[int(j[0])][1], imputed_train[int(j[0])]) > 0.009:
                close.append(int(j[0]))
        if len(close) > 1:
            stop = False
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
        sets = np.zeros((15,300))
        l = 15 if len(close) > 15 else len(close)
        # l = 1
        for k in range(l):
            sets[k] = train_dataset_2[close[k]][1]
        for k in close:
            train_dataset_refined.append((train_dataset_2[k][0], train_dataset_2[k][1], sets, l,\
                np.zeros((num_window, 1)), np.zeros((16, num_period_sample)), np.zeros(num_window)))
    # ######## Add data not in problematic clusters ###########
    for i in range(len(train_dataset_2)):
        if i not in real_added2:
            sets = np.zeros((15,300))
            # sets = np.zeros((1,300))
            sets[0] = (train_dataset_2[i][1])
            train_dataset_refined.append((train_dataset_2[i][0], train_dataset_2[i][1], sets, 1,\
                    np.zeros((num_window, 1)), np.zeros((16, num_period_sample)), np.zeros(num_window)))
    return train_dataset_refined
