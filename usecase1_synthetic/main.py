# 3rd party packages
import torch
import torch.nn as nn
import numpy as np
import random
import os
import pickle

# Project modules
from options import Options
from preprocessing.preprocessor3 import Preprocessor, generate_datasets_pipeline, \
                    data_normalization, process_data
from evaluation.eval import run_downstream_task, run_timing, run_uncertainty 
from model_training.z2n_train import train_z2n
# from preprocessing.preprocessor import Preprocessor

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
    with open("./datasets/pickles_artifact/" + name + ".pickle", "rb") as fin:
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
    pf_list_B5L3= load_dataset('B5L3')
    # pf_list_B5L5= load_dataset('B5L5')
    # pf_list_B3L3F9= load_dataset('B3L3F9')
    # pf_list_dctcp_B7L3F3= load_dataset('dctcp_B7L3F3')
    pf_list_B5L5= []
    pf_list_B3L3F9= []
    pf_list_dctcp_B7L3F3= []

    d_train = np.concatenate((pf_list_B5L3[0:8], pf_list_B5L5[0:8], pf_list_B3L3F9[0:8], pf_list_dctcp_B7L3F3[0:8]))
    d_test = np.concatenate((pf_list_B5L3[8:10], pf_list_B5L5[8:10], pf_list_B3L3F9[8:10], pf_list_dctcp_B7L3F3[8:10]))
    
    for i in range(len(d_train)):
        for j in range (160):
            if j % 2 ==0:
                d_train[i].throughput_data[j][0] = 0
    for i in range(len(d_test)):
        for j in range (160):
            if j % 2 ==0:
                d_test[i].throughput_data[j][0] = 0

    # Generate (coarse-grained, fine-grained) pair
    train_dataset = generate_datasets_pipeline(d_train, window_size =WINDOW_SIZE, window_skip=WINDOW_SKIP, \
                               input_queue_coarsening_factor=COARSE,\
                               output_coarsening_factor=1)
    test_dataset = generate_datasets_pipeline(d_test, window_size =WINDOW_SIZE, window_skip=WINDOW_SKIP, \
                                input_queue_coarsening_factor=COARSE,\
                                output_coarsening_factor=1)
                
    # Minmax normalization
    normalized_train_dataset, normalized_test_dataset, res_queue_max, res_queue_min, res_drop_max, res_drop_min, \
            res_throu_max, res_throu_min = data_normalization(train_dataset, test_dataset)
    throughput_threshold = []
    for i in range(len(d_test)//2):
        for j in range(2*i, 2*i+2):
            th = np.max(d_test[j].throughput_data[64:64+16], axis=1)/res_throu_max
            throughput_threshold.append(th)

    # Prepare training and testing data
    processed_train_dataset, processed_test_dataset = process_data(config, 
                                                train_dataset, test_dataset, include_cca=True)

    if config.task == 'eval_downstream_task':
        print(processed_train_dataset[0][0].shape)
        run_downstream_task(config, test_dataset, train_dataset, len(d_test), throughput_threshold, res_queue_max)
    elif config.task == 'eval_timing':
        run_timing(config, test_dataset, len(d_test), res_queue_max)
    elif config.task == 'eval_uncertainty':
        run_uncertainty(config, processed_test_dataset)
    elif config.task == 'train':
        num_window = int(WINDOW_SIZE / COARSE)
        print(f"Training data size: {len(processed_train_dataset)}")
        print(f"Testing data size: {len(processed_test_dataset)}")
        print(processed_train_dataset[0][0].shape)
        train_z2n(config, processed_train_dataset, processed_test_dataset, 
                    train_dataset, test_dataset, seed)
        
if __name__ == '__main__':
    args = Options().parse()  
    main(args)
