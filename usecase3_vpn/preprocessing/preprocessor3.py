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

def data_normalization(input, output):
    maximum = [1,1,20,20,45121644.8,300439486.24,300439486.24,271209041.2400001,54514632.8,255503311.9599998,130851918.24000002,\
    48800656.03333342, 49132016.48848487, 194.7510060407168, 38199.063448410736, 2081.5199999999677, 1611.0]
    minimum = [0.00000000e+00, 0.00000000e+00, 2.00000000e+00, 2.00000000e+00,\
        0.00000000e+00, 1.24735000e+05, 1.40000000e+01, 0.00000000e+00,\
        0.00000000e+00, 1.40000000e+01, 0.00000000e+00, 1.40000000e+01,\
        0.00000000e+00, 1.16377431e-03, 6.52231841e-02, 4.00000000e+01, 3.90000000e+01]
    minimum_deleted = minimum.copy()
    minimum_deleted = np.delete(minimum_deleted, [0,5,6,7,11,12,14]) 
    maximum_deleted = maximum.copy()
    maximum_deleted = np.delete(maximum_deleted, [0,5,6,7,11,12,14])
    ###### pkt_len #######
    y1_max = 1390
    y1_min = 0
    ###### time #######
    y2_max = 67296475.6
    y2_min = 0
    ####### Normalize
    for i in range(len(output)):
        output[i][0] = (output[i][0] - y1_min) / (y1_max - y1_min)
        output[i][2] = (output[i][2] - y1_min) / (y1_max - y1_min)
        output[i][1] = (output[i][1] - y2_min) / (y2_max - y2_min)
        output[i][3] = (output[i][3] - y2_min) / (y2_max - y2_min)
    for i in range(len(input)):
        for j in range(len(input[0])):
            # if j in [0,1,2,12]:
                input[i][j] = (input[i][j] - minimum_deleted[j]) / (maximum_deleted[j] - minimum_deleted[j])
    return input, output, maximum_deleted, minimum_deleted, y1_min, y1_max, y2_min, y2_max
    