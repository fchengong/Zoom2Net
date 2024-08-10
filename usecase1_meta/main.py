# 3rd party packages
import torch
import torch.nn as nn
import numpy as np
import random
import os

# Project modules
from options import Options
from datasets.preprocessor import Preprocessor, generate_datasets_pipeline, normalization
from evaluation.eval import run_downstream_task, run_zoom_in_factor, run_timing
from model_training.z2n_train import train_z2n
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
    WINDOW_SKIP = config.window_skip
    COARSE = config.zoom_in_factor

    # Load data
    rack_data_train = []
    for i in range(0,150):  
        x = Preprocessor(rackID = i)
        if x.assign == True:
            rack_data_train.append(x)

    rack_data_test = []
    for i in range(150,180): 
        x = Preprocessor(rackID = i)
        if x.assign == True:
            rack_data_test.append(x)

    # Minmax normalization
    ingressBytes_max, connections_max, ingressBytes_min, connections_min, inRxmitBytes_max = \
            normalization(rack_data_train, rack_data_test)
    
    # Prepare training and testing data
    train_dataset = generate_datasets_pipeline(rack_data_train, window_size = WINDOW_SIZE,
        window_skip = WINDOW_SKIP,
        input_coarsening_factor = COARSE,
        output_coarsening_factor = 1)
    
    test_dataset = generate_datasets_pipeline(rack_data_test, window_size = WINDOW_SIZE,
        window_skip = WINDOW_SKIP,
        input_coarsening_factor = COARSE,
        output_coarsening_factor = 1)

    if config.task == 'eval_downstream_task':
        run_downstream_task(config, test_dataset, train_dataset, len(rack_data_test), ingressBytes_max)
    elif config.task == 'eval_timing':
        run_timing(config, test_dataset, len(rack_data_test), ingressBytes_max)
    elif config.task == 'eval_zoom_in_factor':
        run_zoom_in_factor(config, rack_data_test, len(rack_data_test), ingressBytes_max)
    elif config.task == 'train':
        num_window = int(WINDOW_SIZE / COARSE)
        # Aggregate correlated server data
        indexes = []
        for i in range(92):
            a = list(range(i))
            b = list(range(i+1,92))
            indexes.append((a+b))
        train_dataset_processed = []
        for i in range(len(train_dataset)):
            t = train_dataset[i][0][:,[0,2,4,5],:]
            t[:,0,:] = t[:,0,:]/3
            t[:,3,:] = t[:,3,:]/6
            for j in range(92):
                b = (np.sum(t[indexes[j]], axis=0))
                a = np.expand_dims(t[j], axis=0)
                b = np.expand_dims(b, axis=0)/91/2
                d = np.concatenate((a,b))
                train_dataset_processed.append((d, train_dataset[i][1][j]))
        test_dataset_processed = []
        for i in range(len(test_dataset)):
            t = test_dataset[i][0][:,[0,2,4,5],:]
            t[:,0,:] = t[:,0,:]/3
            t[:,3,:] = t[:,3,:]/6
            for j in range(92):
                b = (np.sum(t[indexes[j]], axis=0))
                a = np.expand_dims(t[j], axis=0)
                b = np.expand_dims(b, axis=0)/91/2
                d = np.concatenate((a,b))
                test_dataset_processed.append((d, test_dataset[i][1][j]))

        print(f"Training data size: {len(train_dataset_processed)}")
        print(f"Testing data size: {len(test_dataset_processed)}")
        train_z2n(config, train_dataset_processed, test_dataset_processed, seed)
        
if __name__ == '__main__':
    args = Options().parse()  
    main(args)
