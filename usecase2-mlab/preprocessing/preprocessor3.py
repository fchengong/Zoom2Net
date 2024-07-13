import numpy as np
import glob
import json
import gzip
from torch.utils.data import Dataset

import sys
from typing import Tuple, List
import torch
import numpy as np
from torch.utils.data import Dataset


def data_normalization(train_dataset, test_dataset, flat_pkt, WINDOW):
    maxRTT = 1
    maxBytesRetrans = 1
    maxSndCwnd = 1
    maxCwndGain = 1
    maxelapsetime = 1
    maxsentBytes = 1
    maxRWndLimited = 1
    for n,i in enumerate(train_dataset):
        if np.max(i[0][0]) > maxelapsetime:
            maxelapsetime = np.max(i[0][0])
        if np.max(i[0][1]) > maxRTT:
            maxRTT = np.max(i[0][1])
        if np.max(i[0][2]) > maxBytesRetrans:
            maxBytesRetrans = np.max(i[0][2])
        if np.max(i[0][3]) > maxSndCwnd:
            maxSndCwnd = np.max(i[0][3])
        if np.max(i[0][4]) > maxCwndGain:
            maxCwndGain = np.max(i[0][4])
        if np.max(i[1]) > maxsentBytes:
            maxsentBytes = np.max(i[1])
        if np.max(i[0][8]) > maxRWndLimited:
            maxRWndLimited = np.max(i[0][8])
    maxelapsetime = 3000
    maxsentBytes = np.percentile(flat_pkt, 98)
    for i in range(len(train_dataset)):
        new = np.zeros((7,WINDOW))
        end = np.max(np.cumsum(train_dataset[i][0][0]))//10
        prev = 0
        for n,j in enumerate(((train_dataset[i][0][0])//10).astype(int)):
            if j == 0:
                break
            new[0][prev:j] = train_dataset[i][0][1][n] / maxRTT
            new[1][prev:j] = train_dataset[i][0][2][n] / maxBytesRetrans
            new[2][prev:j] = train_dataset[i][0][3][n] / maxSndCwnd
            new[3][prev:j] = train_dataset[i][0][5][n] / (maxsentBytes)  # /10
            new[4][prev:j] = train_dataset[i][0][6][n] / (maxsentBytes*(j-prev)) # /10
            new[5][prev:j] = train_dataset[i][0][7][n]
            new[6][prev:j] = train_dataset[i][0][8][n] / (maxRWndLimited)
            prev = j
        train_dataset[i][1] /= (maxsentBytes)
        train_dataset[i][0] = new
    for i in range(len(test_dataset)):
        new = np.zeros((7,WINDOW))
        end = np.max(np.cumsum(test_dataset[i][0][0]))//10
        prev = 0
        for n,j in enumerate(((test_dataset[i][0][0])//10).astype(int)):
            if j == 0:
                break
            new[0][prev:j] = test_dataset[i][0][1][n] / maxRTT
            new[1][prev:j] = test_dataset[i][0][2][n] / maxBytesRetrans
            new[2][prev:j] = test_dataset[i][0][3][n] / maxSndCwnd
            new[3][prev:j] = test_dataset[i][0][5][n] / (maxsentBytes)  # /10
            new[4][prev:j] = test_dataset[i][0][6][n] / (maxsentBytes*(j-prev)) # /10
            new[5][prev:j] = test_dataset[i][0][7][n]
            new[6][prev:j] = test_dataset[i][0][8][n] / (maxRWndLimited)
            prev = j
        test_dataset[i][1] /= (maxsentBytes)
        test_dataset[i][0] = new
    return train_dataset, test_dataset, maxRTT, maxBytesRetrans, maxSndCwnd, maxCwndGain, \
            maxelapsetime, maxsentBytes, maxRWndLimited

def process_data(config, train_dataset, test_dataset):
    WINDOW_SIZE = config.window_size
    WINDOW_SKIP = config.window_skip
    coarse = config.zoom_in_factor
    feature_size = 7
    processed_train_dataset = []
    processed_test_dataset = []
    cnt = 0
    for i in train_dataset:
        start = 0
        end = 0
        cnt = 0
        prev = None
        time = np.zeros(feature_size)
        for j in range(WINDOW_SIZE):
            if prev == None:
                prev = i[0][0][j]
            if prev != i[0][0][j]:
                start = end
                time[cnt]=(end)
                cnt += 1
            end += 1
            prev = i[0][0][j]
        processed_train_dataset.append((i[0], i[1], (time)))
        cnt += 1
    cnt = 0
    for i in train_dataset:
        start = 0
        end = 0
        cnt = 0
        prev = None
        time = np.zeros(feature_size)
        for j in range(WINDOW_SIZE):
            if prev == None:
                prev = i[0][0][j]
            if prev != i[0][0][j]:
                start = end
                time[cnt]=(end)
                cnt += 1
            end += 1
            prev = i[0][0][j]
            
        processed_test_dataset.append((i[0], i[1], time))
        cnt += 1
    
    return processed_train_dataset, processed_test_dataset