# 3rd party packages
import torch
import torch.nn as nn
import numpy as np
import random
import os
import pickle
from sklearn.model_selection import train_test_split

# Project modules
from options import Options
from evaluation.eval import run_downstream_task, run_new_features
from model_training.z2n_train import train_z2n
from preprocessing.preprocessor3 import data_normalization

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.determinisdatic = True
    torch.backends.cudnn.benchmark = False
    print ("Seeded everything")


def main(config):
    seed = 78
    set_seed(seed)
    WINDOW_SIZE = config.window_size

    # Load data 
    with open("./datasets/vpn_data/len40_feat17.pickle", "rb") as fin:
        flat_res= pickle.load(fin)
        fin.close()

    input = []
    for i in range(len(flat_res)):
        a = flat_res[i][0]
        a = np.delete(a, [0,5,6,7,11,12,14])
        input.append(a)
    input = np.array(input)
    output = []
    for i in range(len(flat_res)):
        output.append(flat_res[i][1])
    output = np.array(output)

    # Minmax normalization
    normalized_input, normalized_output, maximum_deleted, minimum_deleted, \
                    y1_min, y1_max, y2_min, y2_max= data_normalization(input, output)
    X_train, X_test, Y_train, Y_test = train_test_split(input, output, test_size=0.2, random_state=seed)

    processed_train_dataset = []
    for i in range(len(X_train)):
        processed_train_dataset.append((X_train[i], Y_train[i].flatten(), 0))
    processed_test_dataset = []
    for i in range(len(X_test)):
        processed_test_dataset.append((X_test[i], Y_test[i].flatten()))
        
    if config.task == 'eval_downstream_task':
        run_downstream_task(config, X_train, Y_train, X_test, Y_test, maximum_deleted, \
                            minimum_deleted, y1_min, y1_max, y2_min, y2_max)
    if config.task == 'eval_new_features':
        run_new_features(config, normalized_input, normalized_output, y1_min, y1_max, y2_min, y2_max,\
                        maximum_deleted, minimum_deleted)
    elif config.task == 'train':
        train_z2n(config, processed_train_dataset, processed_test_dataset, seed)
        
if __name__ == '__main__':
    args = Options().parse()  
    main(args)
