import numpy as np
import itertools
import torch
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from evaluation.downstream_task import downstream_task
from model_training.transformer import TSTransformerEncoder
from model_training.utils import inference
from evaluation.brits import model, train_brits
from evaluation.run_inference import load_model

def neighbors(test_index, server, k, train_dataset_knn, test_dataset, indexes):
    # test_index: index into test_dataset
    # server: index into server
    # k: consider first k closest neighbors 
    dist = []
    def myFunc(e):
      return e['dist']
    t = test_dataset[test_index][0]  
    b = (np.sum(t[indexes[server]], axis=0))
    a = np.expand_dims(t[server], axis=0)
    b = np.expand_dims(b, axis=0)/91
    data = np.concatenate((a,b))
    for i in range(len(train_dataset_knn)):     
        d = np.linalg.norm(train_dataset_knn[i][0] - data)
        dist.append({'index': i, 'dist': d})
    dist.sort(key=myFunc)
    x = [i['index'] for i in dist]
    return x[0:k]

def prepare_knn_data(train_dataset, indexes):
    train_dataset_knn = []
    for i in range(len(train_dataset)):
        t = train_dataset[i][0][:,:,:]
        for j in range(92):
            b = (np.sum(t[indexes[j]], axis=0))
            a = np.expand_dims(t[j], axis=0)
            b = np.expand_dims(b, axis=0)/91
            d = np.concatenate((a,b))
            train_dataset_knn.append((d, train_dataset[i][1][j]))
    return train_dataset_knn


def knn(config, test_dataset, train_dataset, rackdata_len, ingressBytes_max):
    WINDOW_SKIP = config.window_skip
    WINDOW_SIZE = config.window_size
    COARSE = config.zoom_in_factor
    num_intervals = len(np.arange(0,2000,WINDOW_SKIP))-1
    num_WINDOW = len(np.arange(0,2000,WINDOW_SIZE))
    skipped = WINDOW_SIZE // WINDOW_SKIP
    if config.compute_baselines == 'False': 
        # Pre-loaded evaluation data
        with open("evaluation/saved_data/res_pred_knn.pickle", "rb") as fin:
            res_pred_knn = pickle.load(fin)
            fin.close()
        res_true_knn = np.zeros((rackdata_len, 92, num_WINDOW, WINDOW_SIZE))
        for server in range(92):
            for i in range(rackdata_len):
                cnt = 0
                for j in range(i*num_intervals, (i+1)*num_intervals):
                    if (j < num_intervals and j % skipped == 0) or \
                    (j >= num_intervals * i and (j - num_intervals * i) % skipped == 0):
                        res_true_knn[i,server,cnt,:] = (test_dataset[j][1][server])
        res_true_knn = np.reshape(res_true_knn, (rackdata_len,92,num_WINDOW*WINDOW_SIZE))
        res_true_knn = np.reshape(res_true_knn, (rackdata_len*92,num_WINDOW*WINDOW_SIZE))
    else:
        # Run KNN froms scrach
        k = 10
        indexes = []
        for i in range(92):
            a = list(range(i))
            b = list(range(i+1,92))
            indexes.append((a+b))
        res_true_knn = np.zeros((rackdata_len, 92, num_WINDOW, WINDOW_SIZE))
        res_pred_knn = np.zeros((rackdata_len, 92, num_WINDOW, WINDOW_SIZE))
        train_dataset_knn = prepare_knn_data(train_dataset, indexes)
        for server in range(92):
            for i in range(rackdata_len):
                cnt = 0
                for j in range(i*num_intervals, (i+1)*num_intervals):
                    if (j < num_intervals and j % skipped == 0) or \
                    (j >= num_intervals * i and (j - num_intervals * i) % skipped == 0):
                        res_true_knn[i,server,cnt,:] = (test_dataset[j][1][server])
                        n = neighbors(j, server, k, train_dataset_knn, test_dataset, indexes)
                        r = [train_dataset_knn[z][1] for z in n]
                        r = np.array(r)
                        x = np.sum(r, axis = 0) / k
                        res_pred_knn[i,server,cnt,:] = (x)
                        cnt += 1
                        
        res_true_knn = np.reshape(res_true_knn, (rackdata_len,92,num_WINDOW*WINDOW_SIZE))
        res_pred_knn = np.reshape(res_pred_knn, (rackdata_len,92,num_WINDOW*WINDOW_SIZE))
        res_true_knn = np.reshape(res_true_knn, (rackdata_len*92,num_WINDOW*WINDOW_SIZE))
        res_pred_knn = np.reshape(res_pred_knn, (rackdata_len*92,num_WINDOW*WINDOW_SIZE))

    res_knn = downstream_task(res_true_knn, res_pred_knn, rackdata_len, ingressBytes_max)

    return res_knn

def plain_transformer(config, test_dataset, rackdata_len, ingressBytes_max):
    WINDOW_SKIP = config.window_skip
    WINDOW_SIZE = config.window_size
    COARSE = config.zoom_in_factor
    model_plain = load_model(config, config.z2n_model_dir, d_model=config.d_model, n_heads=config.n_heads, dim_feedforward=config.dim_feedforward, 
                                zoom_in_factor=config.zoom_in_factor, window_size=config.window_size)
    model_plain.eval()
    indexes = []
    for i in range(92):
        a = list(range(i))
        b = list(range(i+1,92))
        indexes.append((a+b))
    num_intervals = len(np.arange(0,2000,WINDOW_SKIP))-1
    num_WINDOW = len(np.arange(0,2000,WINDOW_SIZE))
    skipped = WINDOW_SIZE // WINDOW_SKIP
    res_true_plain = np.zeros((rackdata_len, 92, num_WINDOW, WINDOW_SIZE))
    res_pred_plain = np.zeros((rackdata_len, 92, num_WINDOW, WINDOW_SIZE))
    for q in range(92):
        for i in range(rackdata_len):
            cnt = 0
            for j in range(i*num_intervals, (i+1)*num_intervals):
                if (j < num_intervals and j % skipped == 0) or \
                (j >= num_intervals * i and (j - num_intervals * i) % skipped == 0):
                    t = test_dataset[j][0][:,[0,2,4,5],:]
                    t[:,0,:] = t[:,0,:]/3
                    t[:,3,:] = t[:,3,:]/6
                    b = (np.sum(t[indexes[q]], axis=0))
                    a = np.expand_dims(t[q], axis=0)
                    b = np.expand_dims(b, axis=0)/91/2
                    d = np.concatenate((a,b))
                    x = inference(model_plain, d, WINDOW_SIZE=WINDOW_SIZE, COARSE=COARSE)[0][0].cpu().numpy()
                    res_true_plain[i,q,cnt,:] = (test_dataset[j][1][q])
                    res_pred_plain[i,q,cnt,:] = (x)
                    cnt += 1
                    
    res_true_plain = np.reshape(res_true_plain, (rackdata_len,92,num_WINDOW*WINDOW_SIZE))
    res_pred_plain = np.reshape(res_pred_plain, (rackdata_len,92,num_WINDOW*WINDOW_SIZE))
    res_true_plain = np.reshape(res_true_plain, (rackdata_len*92,num_WINDOW*WINDOW_SIZE))
    res_pred_plain = np.reshape(res_pred_plain, (rackdata_len*92,num_WINDOW*WINDOW_SIZE))

    res_plain = downstream_task(res_true_plain, res_pred_plain, rackdata_len, ingressBytes_max)

    return res_plain

def iter_imputer(config, test_dataset, rackdata_len, ingressBytes_max):
    WINDOW_SKIP = config.window_skip
    WINDOW_SIZE = config.window_size
    COARSE = config.zoom_in_factor
    num_intervals = len(np.arange(0,2000,WINDOW_SKIP))-1
    num_WINDOW = len(np.arange(0,2000,WINDOW_SIZE))
    skipped = WINDOW_SIZE // WINDOW_SKIP

    if config.compute_baselines == 'False': 
        # Pre-loaded evaluation data
        with open("evaluation/saved_data/res_pred_iter.pickle", "rb") as fin:
            res_pred_iter = pickle.load(fin)
            fin.close()
        res_true_iter = np.zeros((rackdata_len, 92, num_WINDOW, WINDOW_SIZE))
        for q in range(92):
            for i in range(rackdata_len):
                cnt = 0
                for j in range(i*num_intervals, (i+1)*num_intervals):
                    if (j < num_intervals and j % skipped == 0) or \
                    (j >= num_intervals * i and (j - num_intervals * i) % skipped == 0):
                        res_true_iter[i,q,cnt,:] = (test_dataset[j][1][q])
                        cnt += 1
                        # print(cnt)
        res_true_iter = np.reshape(res_true_iter, (rackdata_len,92,num_WINDOW*WINDOW_SIZE))
        res_true_iter = np.reshape(res_true_iter, (rackdata_len*92,num_WINDOW*WINDOW_SIZE))
    else:
        # Run IterImputer from scrach
        iter_imp = IterativeImputer(random_state=0)
        indexes = []
        for i in range(92):
            a = list(range(i))
            b = list(range(i+1,92))
            indexes.append((a+b))
        res_true_iter = np.zeros((rackdata_len, 92, num_WINDOW, WINDOW_SIZE))
        res_pred_iter = np.zeros((rackdata_len, 92, num_WINDOW, WINDOW_SIZE))
        for q in range(92):
            for i in range(rackdata_len):
                cnt = 0
                for j in range(i*num_intervals, (i+1)*num_intervals):
                    if (j < num_intervals and j % skipped == 0) or \
                    (j >= num_intervals * i and (j - num_intervals * i) % skipped == 0):
                        t = test_dataset[j][0][:,[0,2,4,5],:]
                        t[:,0,:] = t[:,0,:]
                        t[:,3,:] = t[:,3,:]
                        r = np.zeros((9,WINDOW_SIZE))
                        b = np.sum(t[indexes[q]], axis=0)/91
                        r[0] = np.nan
                        for z in range(WINDOW_SIZE//COARSE):
                            r[0][z*COARSE] = test_dataset[j][1][q][z*COARSE]
                            r[1][(z+1)*COARSE-1] = t[q,0, z]
                            r[2][(z+1)*COARSE-1] = t[q,1, z]
                            r[3][(z+1)*COARSE-1] = t[q,2, z]
                            r[4][(z+1)*COARSE-1] = t[q,3, z]
                            r[5][(z+1)*COARSE-1] = b[0,z]
                            r[6][(z+1)*COARSE-1] = b[1,z]
                            r[7][(z+1)*COARSE-1] = b[2,z]
                            r[8][(z+1)*COARSE-1] = b[3,z]
                    
                        iter_res = iter_imp.fit_transform(r)
                        res_true_iter[i,q,cnt,:] = (test_dataset[j][1][q])
                        res_pred_iter[i,q,cnt,:] = iter_res[0]
                        cnt += 1
                        # print(cnt)
                        
        res_true_iter = np.reshape(res_true_iter, (rackdata_len,92,num_WINDOW*WINDOW_SIZE))
        res_pred_iter = np.reshape(res_pred_iter, (rackdata_len,92,num_WINDOW*WINDOW_SIZE))
        res_true_iter = np.reshape(res_true_iter, (rackdata_len*92,num_WINDOW*WINDOW_SIZE))
        res_pred_iter = np.reshape(res_pred_iter, (rackdata_len*92,num_WINDOW*WINDOW_SIZE))

    res_iter = downstream_task(res_true_iter, res_pred_iter, rackdata_len, ingressBytes_max)

    return res_iter

def brits(config, test_dataset, train_dataset, rackdata_len, ingressBytes_max):
    if config.compute_baselines == 'False': 
        # Pre-loaded evaluation data
        with open("evaluation/saved_data/res_pred_brits.pickle", "rb") as fin:
            res_pred_brits = pickle.load(fin)
            fin.close()
        with open("evaluation/saved_data/res_true_brits.pickle", "rb") as fin:
            res_true_brits = pickle.load(fin)
            fin.close()
    else:
        # train Brits from scrach
        data_iter_train, data_iter_test = train_brits.prepare_brits_data(config,train_dataset, test_dataset)
        print('start training Brits')
        model = train_brits.train_brits(data_iter_train, data_iter_test, config.window_size)
        res_true_brits, res_pred_brits = train_brits.run_inference(config, test_dataset, model, rackdata_len)
    
    res_brits = downstream_task(res_true_brits, res_pred_brits, rackdata_len, ingressBytes_max)

    return res_brits
