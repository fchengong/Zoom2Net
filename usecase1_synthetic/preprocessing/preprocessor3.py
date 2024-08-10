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

def np_loadnpz(file: str):
    npz_file = np.load(file)
    return npz_file[npz_file.files[0]]

class Preprocessor:
    def __init__(self,
        data_path: str = "",
        target_rate: int = 1000,
        data_ends_at: int = 1e7,
        run_id: int = 0,
        nport: int = 40,
        nqueue_per_port: int = 2):
            """
            Initialize the preprocessor.

            Parameters
            ----------------------------------
            data_path: str
                The folder to look for data files. Data file names are hardcoded.

            target_rate: int
                The target sampling rate we want to impute to, in the unit of us.
                This will be the finest data granularity we consider. Referred to as a "time unit".
                Default is 1000, which means 1 time unit = 1ms.

            data_ends_at: int
                Ignore input data after this number of microseconds. This may represent an inactive period at the end of data.
                Use -1 to ignore.
            
            run_id: int
                The ID of the run. Used to identify when there are multiple simulation runs.
                Directly combining multiple runs of data is not helpful because the windowing procedure will window across different runs, which do not make sense.
                To get more training/testing samples from multiple runs, initialize preprocessor with different runs, get sample sets and combine them.
            """
            self.data_path = data_path
            self.target_rate = target_rate
            
            # Test input data length.
            test = os.path.join(data_path,f"run-{run_id}-leaf-0-drop-0-0.npz")
            test_arr = np_loadnpz(test)

            data_origin_size = test_arr.size
            data_cut_size = data_ends_at if (data_ends_at > 0 and data_ends_at < data_origin_size) else data_origin_size
            self.data_size = data_cut_size
            del test_arr
            del test

            nqueue = nport * nqueue_per_port
            data_target_size = int(data_cut_size/target_rate)
            self.queue_data = np.zeros((160, data_target_size))
            #self.threshold_data = np.zeros((160, data_target_size))
            self.throughput_in_data = np.zeros((160, data_target_size)) 
            self.drop_data = np.zeros((160, data_target_size))
            self.throughput_data = np.zeros((160, data_target_size))

            for leaf in range(1):
                for port in range(40):
                    for queue in range(2):
                        index = leaf * nqueue + port * nqueue_per_port + queue
                        print(f"\rLoading Leaf {leaf}, port {port+1}/40, buffer {queue+1}/2. (Index {index})", end='', flush=True)

                        # Queue length data, max pooling
                        queue_fname = os.path.join(data_path,f"run-{run_id}-leaf-{leaf}-queuelength-{port}-{queue}.npz")
                        origin = np_loadnpz(queue_fname)

                        if(origin.size < data_cut_size):
                            raise ValueError("Input data is not sufficiently long.")
                        
                        target_granularity_arr = np.zeros(int(data_cut_size/target_rate))
                        for i in np.arange(0, int(data_cut_size/target_rate)):
                            target_granularity_arr[i] = np.max(origin[i*target_rate:(i+1)*target_rate])
                            #target_granularity_arr[i] = np.average(origin[i*target_rate:(i+1)*target_rate])
                        self.queue_data[index] = target_granularity_arr
                        del origin
                        gc.collect()

                        # Drop rate data, max pooling.
                        drop_fname = os.path.join(data_path,f"run-{run_id}-leaf-{leaf}-drop-{port}-{queue}.npz")
                        origin: np.ndarray = np_loadnpz(drop_fname)

                        if(origin.size < data_cut_size):
                            raise ValueError("Input data is not sufficiently long.")

                        target_granularity_arr = np.zeros(int(data_cut_size/target_rate))
                        for i in np.arange(0, int(data_cut_size/target_rate)):
                            target_granularity_arr[i] = np.max(origin[i*target_rate:(i+1)*target_rate])

                        self.drop_data[index] = target_granularity_arr
                        del origin
                        gc.collect()

                        # Throughput data
                        throughput_fname = os.path.join(data_path,f"run-{run_id}-leaf-{leaf}-threshold-{port}-{queue}.npz")
                        origin: np.ndarray = np_loadnpz(throughput_fname)

                        if(origin.size < data_cut_size):
                            raise ValueError("Input data is not sufficiently long.")

                        target_granularity_arr = np.zeros(int(data_cut_size/target_rate))
                        for i in np.arange(0, int(data_cut_size/target_rate)):
                            target_granularity_arr[i] = np.average(origin[i*target_rate:(i+1)*target_rate])
                        
                        self.throughput_data[index] = target_granularity_arr
                        del origin
                        gc.collect()

            self.data_size = data_target_size
            self.dimensions = nqueue

    @staticmethod
    def maxsample(origin_data, coarsening_factor):
        sampled = np.zeros(int(len(origin_data)/coarsening_factor))
        for i in np.arange(0, int(len(origin_data)/coarsening_factor)):
            sampled[i] = np.max(origin_data[i*coarsening_factor:(i+1)*coarsening_factor])
        return sampled

    @staticmethod
    def periodicavg(origin_data, sample_period):
        sampled = np.zeros(int(len(origin_data)/sample_period))
        for i in range(int(len(origin_data)/sample_period)):
            sampled[i] = np.sum(origin_data[i*sample_period:(i+1)*sample_period])/sample_period

        return sampled

    @staticmethod
    def periodicavg_offset(origin_data, sample_period, offset):
        sampled = np.zeros(int(len(origin_data)/sample_period))
        for i in range(int(len(origin_data)/sample_period)):
            #sampled[i] = np.sum(origin_data[i*sample_period:(i+1)*sample_period])/sample_period
            if (i+1)*sample_period + offset > len(origin_data):
                end = len(origin_data)
                cnt = len(origin_data) - (i * sample_period + offset)
            else:
                end = (i+1)*sample_period + offset
                cnt = sample_period
            if i == 0:
                start = 0
                cnt += offset
            else:
                start = i * sample_period + offset
            sampled[i] = np.sum(origin_data[start:end])/cnt

        return sampled

    @staticmethod
    def periodicsum(origin_data, sample_period):
        sampled = np.zeros(int(len(origin_data)/sample_period))
        for i in range(int(len(origin_data)/sample_period)):
            sampled[i] = np.sum(origin_data[i*sample_period:(i+1)*sample_period])

        return sampled
    
    @staticmethod
    def periodicsum_offset(origin_data, sample_period, offset):
        sampled = np.zeros(int(len(origin_data)/sample_period))
        for i in range(int(len(origin_data)/sample_period)):
            if (i+1)*sample_period + offset > len(origin_data):
                end = len(origin_data)
            else:
                end = (i+1)*sample_period + offset
            if i == 0:
                start = 0
            else:
                start = i*sample_period+offset
            sampled[i] = np.sum(origin_data[start:end])
        return sampled

    @staticmethod
    def periodicsample_indexed(origin_data, sample_period, offset):
        index = (np.arange(0, origin_data.size, sample_period) + offset) % len(origin_data)
        sampled = origin_data[index]
        return sampled, index

    def get_windowed_samples_output_pipeline(self,
        window_size: int = 10000,
        window_skip: int = 1000,
        portion_training: float = 0.8,
        input_queue_coarsening_factor: int = 1000,
        output_coarsening_factor: int = 100,
        pick_method = "random"
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
        processed_data = []
        current_window_start: int = 0
        while(current_window_start + window_size <= self.data_size):
            inputs = np.zeros((16, 4, int(window_size/output_coarsening_factor)))
            outputs1 = np.zeros((16, int(window_size/output_coarsening_factor)))
            interval = int(input_queue_coarsening_factor / output_coarsening_factor)
            for j in range(16):
                i = 64 + j 
                qlen_cut = self.queue_data[i, current_window_start:current_window_start + window_size]
                if i % 2 == 0:
                    throughput_cut = self.throughput_data[i, current_window_start:current_window_start + window_size]/8*1000 + self.throughput_data[i+1, current_window_start:current_window_start + window_size]/8*1000
                    drop_cut = self.drop_data[i, current_window_start:current_window_start + window_size] + self.drop_data[i+1, current_window_start:current_window_start + window_size]
                else:
                    throughput_cut = self.throughput_data[i, current_window_start:current_window_start + window_size]/8*1000 + self.throughput_data[i-1, current_window_start:current_window_start + window_size]/8*1000
                    drop_cut = self.drop_data[i, current_window_start:current_window_start + window_size] + self.drop_data[i-1, current_window_start:current_window_start + window_size]

                queue_sampled = Preprocessor.maxsample(qlen_cut, input_queue_coarsening_factor)
                queue = np.zeros(int(window_size/output_coarsening_factor))
                for index,v in enumerate(queue_sampled):
                    queue[index * interval : (index + 1) * interval]= v
                offset = 0
                queue2_sampled, index = Preprocessor.periodicsample_indexed(qlen_cut, int(input_queue_coarsening_factor),offset)
                queue2 = np.zeros(int(window_size/output_coarsening_factor))
                for ind, v in enumerate(index):
                        queue2[int(v / output_coarsening_factor)]= queue2_sampled[ind]
                
                drop_sampled = Preprocessor.periodicsum_offset(drop_cut, input_queue_coarsening_factor, offset)
                drop = np.zeros(int(window_size/output_coarsening_factor))
                for index,v in enumerate(drop_sampled):
                    if (index + 1) * interval + offset > len(queue):
                        end = len(queue)
                    else:
                        end = (index + 1) * interval + offset
                    if index == 0:
                        start = 0
                    else:
                        start = index * interval + offset
                    drop[start : end] = v

                throughput_sampled = Preprocessor.periodicavg_offset(throughput_cut, int(input_queue_coarsening_factor), offset)
                throughput = np.zeros(int(window_size/output_coarsening_factor))
                for index,v in enumerate(throughput_sampled):
                    if (index + 1) * interval + offset > len(queue):
                            end = len(queue)
                    else:
                        end = (index + 1) * interval + offset
                    if index == 0:
                        start = 0
                    else:
                        start = index * interval + offset
                    throughput[start : end] = v

                inputs[j, 0] = queue
                inputs[j, 3] = queue2
                inputs[j, 1] = drop
                inputs[j, 2] = throughput
                queue_output1 = Preprocessor.maxsample(qlen_cut, output_coarsening_factor)
                
                outputs1[j] = queue_output1
            processed_data.append([inputs, outputs1])

            current_window_start = current_window_start + window_skip
        return processed_data

QLEN_MAX = 20000
DROP_MAX = 20000
THRES_MAX = 1000
THROUGHPUT_MAX = 2e6

class Z2NNormalizedInformedPipelineDataSet(Dataset):
    def __init__(self, data: List[List[np.ndarray]]):
        self.data = []
        for d in data:
            self.data.append(d)
    def __len__(self):
        return len(self.data)

    # Data, label.
    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

def generate_datasets_pipeline(
    preprocessors: List[Preprocessor],
    window_size: int,
    window_skip: int,
    input_queue_coarsening_factor: int,
    output_coarsening_factor: int,
    train_ratio = 0.8
) -> Tuple[Z2NNormalizedInformedPipelineDataSet, Z2NNormalizedInformedPipelineDataSet]:
    all = []
    for pf in preprocessors:
        d = pf.get_windowed_samples_output_pipeline(window_size=window_size, window_skip=window_skip, 
        input_queue_coarsening_factor=input_queue_coarsening_factor, 
        output_coarsening_factor=output_coarsening_factor, 
        portion_training=train_ratio)
        all = all + d
    return Z2NNormalizedInformedPipelineDataSet(all)

def data_normalization(train_dataset, test_dataset):
    res_queue_max = []
    res_queue_min = []
    res_drop_max = []
    res_drop_min = []
    res_throu_max = []
    res_throu_min = []
    for queue in range(16):
        maxqueues = []
        minqueues = []
        for data in range(len(train_dataset)):
            maxqueues.append(np.amax(train_dataset[data][0][queue][0]))
            minqueues.append(np.amin(train_dataset[data][0][queue][0]))
            
        maximum_queues = 1 if np.amax(maxqueues) == 0 else np.amax(maxqueues)
        minimum_queues = np.amin(minqueues)
        res_queue_max.append(maximum_queues)
        res_queue_min.append(minimum_queues)
        for i in range(len(train_dataset)):
            train_dataset[i][0][queue][0] = (train_dataset[i][0][queue][0] - minimum_queues) / (maximum_queues - minimum_queues)
            train_dataset[i][0][queue][3] = (train_dataset[i][0][queue][3] - minimum_queues) / (maximum_queues - minimum_queues)
            train_dataset[i][1][queue] = (train_dataset[i][1][queue] - minimum_queues) / (maximum_queues - minimum_queues)
        for i in range(len(test_dataset)):
            test_dataset[i][0][queue][0] = (test_dataset[i][0][queue][0] - minimum_queues) / (maximum_queues - minimum_queues)
            test_dataset[i][0][queue][3] = (test_dataset[i][0][queue][3] - minimum_queues) / (maximum_queues - minimum_queues)
            test_dataset[i][1][queue] = (test_dataset[i][1][queue] - minimum_queues) / (maximum_queues - minimum_queues)

        maxdrops = []
        mindrops = []
        for data in range(len(train_dataset)):
            maxdrops.append(np.amax(train_dataset[data][0][queue][1]))
            mindrops.append(np.amin(train_dataset[data][0][queue][1]))

        maximum_drop = 1 if np.amax(maxdrops) == 0 else np.amax(maxdrops)
        minimum_drop = np.amin(mindrops)
        res_drop_max.append(maximum_drop)
        res_drop_min.append(minimum_drop)
        
        for i in range(len(train_dataset)):
            train_dataset[i][0][queue][1] = (train_dataset[i][0][queue][1] - minimum_drop) / (maximum_drop - minimum_drop)    
        for i in range(len(test_dataset)):
            test_dataset[i][0][queue][1] = (test_dataset[i][0][queue][1] - minimum_drop) / (maximum_drop - minimum_drop)
     
        maxthroughput = []
        minthroughput = []
        for data in range(len(train_dataset)):
            maxthroughput.append(np.amax(train_dataset[data][0][queue][2]))
            minthroughput.append(np.amin(train_dataset[data][0][queue][2]))
        for data in range(len(test_dataset)):
            maxthroughput.append(np.amax(test_dataset[data][0][queue][2]))
            minthroughput.append(np.amin(test_dataset[data][0][queue][2]))
            
        maximum_throughput = 1 if np.amax(maxthroughput) == 0 else np.amax(maxthroughput)
        minimum_throughput = np.amin(minthroughput)
        res_throu_max.append(maximum_throughput)
        res_throu_min.append(minimum_throughput)
        for i in range(len(train_dataset)):
            train_dataset[i][0][queue][2] = (train_dataset[i][0][queue][2] - minimum_throughput) / (maximum_throughput - minimum_throughput)
        for i in range(len(test_dataset)):
            test_dataset[i][0][queue][2] = (test_dataset[i][0][queue][2] - minimum_throughput) / (maximum_throughput - minimum_throughput)
    return train_dataset, test_dataset, res_queue_max, res_queue_min, res_drop_max, res_drop_min, \
            res_throu_max, res_throu_min

def convert(data):
    a = np.zeros((8,6,6))
    for i in range(8):
        for j in range(6):
            a[i,0,j] = data[i*2][0][j*50]
            a[i,1,j] = data[i*2][3][j*50]
            a[i,2,j] = data[i*2+1][0][j*50] *2
            a[i,3,j] = data[i*2+1][3][j*50]
            a[i,4,j] = data[i*2][1][j*50]
            a[i,5,j] = data[i*2][2][j*50]
    return a

def convert_odd(data):
    a = np.zeros((8,6,6))
    for i in range(8):
        for j in range(6):
            a[i,0,j] = data[i*2][0][j*50]
            a[i,1,j] = data[i*2][3][j*50]
            a[i,2,j] = data[i*2+1][0][j*50] *2
            a[i,3,j] = data[i*2+1][3][j*50]
            a[i,4,j] = data[i*2][1][j*50]
            a[i,5,j] = data[i*2][2][j*50]
    return a

def convert_even(data):
    a = np.zeros((8,6,6))
    for i in range(8):
        for j in range(6):
            a[i,0,j] = data[i*2][0][j*50] * 10
            a[i,1,j] = data[i*2][3][j*50]
            a[i,2,j] = data[i*2+1][0][j*50]
            a[i,3,j] = data[i*2+1][3][j*50]
            a[i,4,j] = data[i*2][1][j*50]
            a[i,5,j] = data[i*2][2][j*50]
    return a

def process_data(config, train_dataset, test_dataset, include_cca):
    WINDOW_SIZE = config.window_size
    WINDOW_SKIP = config.window_skip
    coarse = config.zoom_in_factor
    num_window = int(WINDOW_SIZE / coarse)   # 6
    num_period_sample = len(np.arange(0, WINDOW_SIZE, int(coarse/2)))
    num_queue = 1
    processed_train_dataset = []
    processed_test_dataset = []
    indexes = []
    for i in range(8):
        a = list(range(i))
        b = list(range(i+1,8))
        indexes.append((a+b))

    for i in range(len(train_dataset)):
        converted = convert(train_dataset[i][0])
        
        # Note DCTCP or Cubic CCA
        if i < len(train_dataset)/2:
            c = np.zeros((1,6))
        else:
            c = np.ones((1,6))
        
        # Aggregate features among other ports
        for j in range(8):
            b = (np.sum(converted[indexes[j]], axis=0))/7
            if include_cca:
                a = np.expand_dims(np.concatenate((converted[j],c)), axis=0)
                b = np.expand_dims(np.concatenate((b,c)), axis=0)
            else:
                a = np.expand_dims(converted[j], axis=0)
                b = np.expand_dims(b, axis=0)
            d = np.concatenate((a,b))
            
            processed_train_dataset.append((d, train_dataset[i][1][2*j+1],np.zeros((num_window)),1,\
                        np.zeros((num_window, num_queue)), np.zeros((16, num_period_sample)), np.zeros(num_window)))

    for i in range(len(test_dataset)):
        converted = convert(test_dataset[i][0])
        if i < len(test_dataset)/2:
            c = np.zeros((1,6))
        else:
            c = np.ones((1,6))
        for j in range(8):
            b = (np.sum(converted[indexes[j]], axis=0))/7
            if include_cca:
                a = np.expand_dims(np.concatenate((converted[j],c)), axis=0)
                b = np.expand_dims(np.concatenate((b,c)), axis=0)
            else:
                a = np.expand_dims(converted[j], axis=0)
                b = np.expand_dims(b, axis=0)
            d = np.concatenate((a,b))
            
            processed_test_dataset.append((d, test_dataset[i][1][2*j+1]))
    
    return processed_train_dataset, processed_test_dataset