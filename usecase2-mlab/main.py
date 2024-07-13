# 3rd party packages
import torch
import torch.nn as nn
import numpy as np
import random
import os
import pickle

# Project modules
from options import Options
from evaluation.eval import run_downstream_task 
from model_training.z2n_train import train_z2n
from preprocessing.preprocessor3 import data_normalization, process_data
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.determinisdatic = True
    torch.backends.cudnn.benchmark = False
    print ("Seeded everything")

def load_dataset(name):
    with open("./datasets/mlab_data/" + name + ".pickle", "rb") as fin:
        data= pickle.load(fin)
        fin.close()
    return data

def main(config):
    seed = 78
    set_seed(seed)
    WINDOW_SIZE = config.window_size
    WINDOW_SKIP = config.window_skip
    COARSE = config.zoom_in_factor

    # Load data 
    data_3s= load_dataset('data300fine_new_3s')
    data_3to6s= load_dataset('data300fine_new_3to6s')
    data_6to9s= load_dataset('data300fine_new_6to9s')
    AppLimited_3s= load_dataset('AppLimited_new_3s')
    RWndLimited_3s= load_dataset('RWndLimited_new_3s')
    AppLimited_3to6s= load_dataset('AppLimited_new_3to6s')
    RWndLimited_3to6s= load_dataset('RWndLimited_new_3to6s')
    AppLimited_6to9s= load_dataset('AppLimited_new_6to9s')
    RWndLimited_6to9s= load_dataset('RWndLimited_new_6to9s')
    for i in range(len(data_3to6s)):
        data_3to6s[i][0][0][data_3to6s[i][0][0] < 0] = 0
    for i in range(len(data_6to9s)):
        data_6to9s[i][0][0][data_6to9s[i][0][0] < 0] = 0

    data_3s_train = int(len(data_3s)*0.8)
    data_3s_test = len(data_3s) - data_3s_train
    data_3to6s_train = int(len(data_3to6s)*0.8)
    data_3to6s_test = len(data_3to6s) - data_3to6s_train
    data_6to9s_train = int(len(data_6to9s)*0.8)
    data_6to9s_test = len(data_6to9s) - data_6to9s_train
    # data_3to6s_train = 0
    # data_3to6s_test = 0
    # data_6to9s_train = 0
    # data_6to9s_test = 0
    train_dataset = []

    print(data_3s_train, data_3s_test, data_3to6s_train, data_3to6s_test, data_6to9s_train, data_6to9s_test)
    # Training data
    for i in range(data_3s_train):
        data_3s[i][0] = np.append(data_3s[i][0],np.expand_dims(AppLimited_3s[i], axis=0), axis=0)
        data_3s[i][0] = np.append(data_3s[i][0],np.expand_dims(RWndLimited_3s[i], axis=0), axis=0)
        train_dataset.append(data_3s[i])
    for i in range(data_3to6s_train):
        data_3to6s[i][0] = np.append(data_3to6s[i][0],np.expand_dims(AppLimited_3to6s[i], axis=0), axis=0)
        data_3to6s[i][0] = np.append(data_3to6s[i][0],np.expand_dims(RWndLimited_3to6s[i], axis=0), axis=0)
        train_dataset.append(data_3to6s[i])
    for i in range(data_6to9s_train):
        data_6to9s[i][0] = np.append(data_6to9s[i][0],np.expand_dims(AppLimited_6to9s[i], axis=0), axis=0)
        data_6to9s[i][0] = np.append(data_6to9s[i][0],np.expand_dims(RWndLimited_6to9s[i], axis=0), axis=0)
        train_dataset.append(data_6to9s[i])
    test_dataset = []
    # Testing data
    for i in range(data_3s_train,len(data_3s)):
        data_3s[i][0] = np.append(data_3s[i][0],np.expand_dims(AppLimited_3s[i], axis=0), axis=0)
        data_3s[i][0] = np.append(data_3s[i][0],np.expand_dims(RWndLimited_3s[i], axis=0), axis=0)
        test_dataset.append(data_3s[i])
    for i in range(data_3to6s_train,len(data_3to6s)):
        data_3to6s[i][0] = np.append(data_3to6s[i][0],np.expand_dims(AppLimited_3to6s[i], axis=0), axis=0)
        data_3to6s[i][0] = np.append(data_3to6s[i][0],np.expand_dims(RWndLimited_3to6s[i], axis=0), axis=0)
        test_dataset.append(data_3to6s[i])
    for i in range(data_6to9s_train,len(data_6to9s)):
        data_6to9s[i][0] = np.append(data_6to9s[i][0],np.expand_dims(AppLimited_6to9s[i], axis=0), axis=0)
        data_6to9s[i][0] = np.append(data_6to9s[i][0],np.expand_dims(RWndLimited_6to9s[i], axis=0), axis=0)
        test_dataset.append(data_6to9s[i])

    pkt = []
    for i in train_dataset:
        pkt.append(i[1])
    flat_pkt = [item for sublist in pkt for item in sublist]

    # Minmax normalization
    normalized_train_dataset, normalized_test_dataset, rmaxRTT, maxBytesRetrans, maxSndCwnd, maxCwndGain, \
            maxelapsetime, maxsentBytes, maxRWndLimited = data_normalization(train_dataset, \
                test_dataset, flat_pkt, config.window_size)

    # # Prepare training and testing data
    processed_train_dataset, processed_test_dataset = process_data(config, 
                                                train_dataset, test_dataset)

    if config.task == 'eval_downstream_task':
        print(f"Training data size: {len(processed_train_dataset)}")
        print(f"Testing data size: {len(processed_test_dataset)}")
        print(processed_train_dataset[0][0].shape)
        run_downstream_task(config, processed_test_dataset, processed_train_dataset, maxsentBytes,\
                data_3s_train, data_3s_test, data_3to6s_train, data_3to6s_test, data_6to9s_train, data_6to9s_test)
    elif config.task == 'train':
        print(f"Training data size: {len(processed_train_dataset)}")
        print(f"Testing data size: {len(processed_test_dataset)}")
        print(processed_train_dataset[0][0].shape)
        train_z2n(config, processed_train_dataset, processed_test_dataset, 
                    train_dataset, test_dataset, seed)
        
if __name__ == '__main__':
    args = Options().parse()  
    main(args)
