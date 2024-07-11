import numpy as np
import glob
import json
import gzip
from torch.utils.data import Dataset

class Preprocessor:
    def __init__(self,rackID):
        self.ingressBytes = np.zeros((92, 2000))
        self.egressBytes = np.zeros((92, 2000))
        self.inRxmitBytes = np.zeros((92, 2000))
        self.outRxmitBytes = np.zeros((92, 2000))
        self.inCongestionBytes = np.zeros((92, 2000))
        self.connections = np.zeros((92, 2000))
        self.assign = False
        self.data_size = 2000

        searchname = './datasets/Millisampler/day1-h1-zip/rackId_' + str(rackID) + '_hostId_*.txt.gz'
        FilenamesList = glob.glob(searchname)
        hostnum = 0
        if len(FilenamesList) != 0:
            self.assign = True
            for filename in FilenamesList:
                f=gzip.open(filename,'rb')
                file_content=f.read()
                js = json.loads(file_content)
                try:
                    ingressBytes = js['ingressBytes']
                except:
                    continue
                egressBytes = js['egressBytes']
                inRxmitBytes = js['inRxmitBytes']
                outRxmitBytes = js['outRxmitBytes']
                inCongestionBytes = js['inCongestionExperiencedBytes']
                connections = js['connections']
                self.ingressBytes[hostnum, :len(ingressBytes)] = ingressBytes
                self.egressBytes[hostnum, :len(egressBytes)] = egressBytes
                self.inRxmitBytes[hostnum, :len(inRxmitBytes)] = inRxmitBytes
                self.outRxmitBytes[hostnum, :len(outRxmitBytes)] = outRxmitBytes
                self.inCongestionBytes[hostnum, :len(inCongestionBytes)] = inCongestionBytes
                self.connections[hostnum, :len(connections)] = connections
                hostnum += 1

    @staticmethod
    def periodicsum(origin_data, sample_period):
        sampled = np.zeros(int(len(origin_data)/sample_period))
        for i in range(int(len(origin_data)/sample_period)):
            if (i+1)*sample_period > len(origin_data):
                end = len(origin_data)
            else:
                end = (i+1)*sample_period
            if i == 0:
                start = 0
            else:
                start = i*sample_period
            sampled[i] = np.sum(origin_data[start:end])

        return sampled
        
    # Generate coarse-grained features
    def get_samples(self,
        window_size: int = 10000,
        window_skip: int = 1000,
        input_coarsening_factor: int = 1000,
        output_coarsening_factor: int = 100):
        sampled_data = []
        current_window_start: int = 0
        input_size = int(window_size/input_coarsening_factor)
        while(current_window_start + window_size <= self.data_size):
            inputs = np.zeros((92, 6, input_size))
            outputs = np.zeros((92, int(window_size/output_coarsening_factor)))
            for i in range(92):
                ingressBytes_cut = self.ingressBytes[i, current_window_start:current_window_start + window_size]
                egressBytes_cut = self.egressBytes[i, current_window_start:current_window_start + window_size]
                inRxmitBytes_cut = self.inRxmitBytes[i, current_window_start:current_window_start + window_size]
                outRxmitBytes_cut = self.outRxmitBytes[i, current_window_start:current_window_start + window_size]
                inCongestionBytes_cut = self.inCongestionBytes[i, current_window_start:current_window_start + window_size]
                connections_cut = self.connections[i, current_window_start:current_window_start + window_size]
                
                # ingressBytes
                ingressBytes_sampled = Preprocessor.periodicsum(ingressBytes_cut, input_coarsening_factor)
                ingressBytes = np.zeros(input_size)
                ingressBytes = ingressBytes_sampled
                
                # egressBytes 
                egressBytes_sampled = Preprocessor.periodicsum(egressBytes_cut, input_coarsening_factor)
                egressBytes = np.zeros(input_size)
                egressBytes = egressBytes_sampled

                # inRxmitBytes 
                inRxmitBytes_sampled = Preprocessor.periodicsum(inRxmitBytes_cut, input_coarsening_factor)
                inRxmitBytes = np.zeros(input_size)
                inRxmitBytes = inRxmitBytes_sampled

                # outRxmitBytes 
                outRxmitBytes_sampled = Preprocessor.periodicsum(outRxmitBytes_cut, input_coarsening_factor)
                outRxmitBytes = np.zeros(input_size)
                outRxmitBytes = outRxmitBytes_sampled

                # inCongestionBytes 
                inCongestionBytes_sampled = Preprocessor.periodicsum(inCongestionBytes_cut, input_coarsening_factor)
                inCongestionBytes = np.zeros(input_size)
                inCongestionBytes = inCongestionBytes_sampled

                # connections 
                connections_sampled = Preprocessor.periodicsum(connections_cut, input_coarsening_factor)
                connections = np.zeros(input_size)
                connections = connections_sampled

                inputs[i, 0] = ingressBytes
                inputs[i, 1] = egressBytes
                inputs[i, 2] = inRxmitBytes
                inputs[i, 3] = outRxmitBytes
                inputs[i, 4] = inCongestionBytes
                inputs[i, 5] = connections

                outputs[i] = ingressBytes_cut

            sampled_data.append([inputs, outputs])

            current_window_start = current_window_start + window_skip
        return sampled_data

class Data(Dataset):
    def __init__(self, data):
        self.data = []
        for d in data:
            self.data.append(d)
    def __len__(self):
        return len(self.data)

    # Data, label.
    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

def generate_datasets_pipeline(
    preprocessors,
    window_size: int,
    window_skip: int,
    input_coarsening_factor: int,
    output_coarsening_factor: int):
    alldata = []
    for pf in preprocessors:
        d = Preprocessor.get_samples(pf, window_size,
        window_skip,
        input_coarsening_factor,
        output_coarsening_factor)    
        alldata = alldata + d
    return Data(alldata)


def normalization(rack_data_train, rack_data_test):
    ingressBytes_max = 0
    ingressBytes_min = float('inf')
    egressBytes_max = 0
    egressBytes_min = float('inf')
    inRxmitBytes_max = 0
    inRxmitBytes_min = float('inf')
    outRxmitBytes_max = 0
    outRxmitBytes_min = float('inf')
    inCongestionBytes_max = 0
    inCongestionBytes_min = float('inf')
    connections_max = 0
    connections_min = float('inf')
    for d in rack_data_train:
        if np.max(d.ingressBytes) > ingressBytes_max:
            ingressBytes_max = np.max(d.ingressBytes)
        if np.max(d.egressBytes) > egressBytes_max:
            egressBytes_max = np.max(d.egressBytes)
        if np.max(d.inRxmitBytes) > inRxmitBytes_max:
            inRxmitBytes_max = np.max(d.inRxmitBytes)
        if np.max(d.outRxmitBytes) > outRxmitBytes_max:
            outRxmitBytes_max = np.max(d.outRxmitBytes)
        if np.max(d.inCongestionBytes) > inCongestionBytes_max:
            inCongestionBytes_max = np.max(d.inCongestionBytes)
        if np.max(d.connections) > connections_max:
            connections_max = np.max(d.connections)
        if np.min(d.ingressBytes) < ingressBytes_min:
            ingressBytes_min = np.min(d.ingressBytes)
        if np.min(d.egressBytes) < egressBytes_min:
            egressBytes_min = np.min(d.egressBytes)
        if np.min(d.inRxmitBytes) < inRxmitBytes_min:
            inRxmitBytes_min = np.min(d.inRxmitBytes)
        if np.min(d.outRxmitBytes) < outRxmitBytes_min:
            outRxmitBytes_min = np.min(d.outRxmitBytes)
        if np.min(d.inCongestionBytes) < inCongestionBytes_min:
            inCongestionBytes_min = np.min(d.inCongestionBytes)
        if np.min(d.connections) < connections_min:
            connections_min = np.min(d.connections)
    for i in range(len(rack_data_train)):
        rack_data_train[i].ingressBytes = (rack_data_train[i].ingressBytes-ingressBytes_min) / (ingressBytes_max-ingressBytes_min)
        rack_data_train[i].egressBytes = (rack_data_train[i].egressBytes-ingressBytes_min) / (ingressBytes_max-ingressBytes_min)
        rack_data_train[i].inRxmitBytes = (rack_data_train[i].inRxmitBytes-ingressBytes_min) / (ingressBytes_max-ingressBytes_min)
        rack_data_train[i].outRxmitBytes = (rack_data_train[i].outRxmitBytes-ingressBytes_min) / (ingressBytes_max-ingressBytes_min)
        rack_data_train[i].inCongestionBytes = (rack_data_train[i].inCongestionBytes-ingressBytes_min) / (ingressBytes_max-ingressBytes_min)
        rack_data_train[i].connections = (rack_data_train[i].connections-connections_min) / (connections_max-connections_min)
    for i in range(len(rack_data_test)):
        rack_data_test[i].ingressBytes = (rack_data_test[i].ingressBytes-ingressBytes_min) / (ingressBytes_max-ingressBytes_min)
        rack_data_test[i].egressBytes = (rack_data_test[i].egressBytes-ingressBytes_min) / (ingressBytes_max-ingressBytes_min)
        rack_data_test[i].inRxmitBytes = (rack_data_test[i].inRxmitBytes-ingressBytes_min) / (ingressBytes_max-ingressBytes_min)
        rack_data_test[i].outRxmitBytes = (rack_data_test[i].outRxmitBytes-ingressBytes_min) / (ingressBytes_max-ingressBytes_min)
        rack_data_test[i].inCongestionBytes = (rack_data_test[i].inCongestionBytes-ingressBytes_min) / (ingressBytes_max-ingressBytes_min)
        rack_data_test[i].connections = (rack_data_test[i].connections-connections_min) / (connections_max-connections_min)

    return ingressBytes_max, connections_max, ingressBytes_min, connections_min, inRxmitBytes_max