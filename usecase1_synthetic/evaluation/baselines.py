import numpy as np
import itertools
import torch
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from evaluation.downstream_task import downstream_task
from model_training.transformer import TSTransformerEncoder
from model_training.utils import inference, inference_withoutCCA, convert_even_plain, convert_odd_plain
from evaluation.brits import model, train_brits
from evaluation.run_inference import load_model
from preprocessing.preprocessor3 import convert


def neighbors(test_index, k, feature_ports, indexes, train_dataset_knn, test_dataset):
    # test_index: index into test_dataset
    # feature_ports: which queue to be imputed
    # k: consider first k closest neighbors 
    # train_dataset_knn: training data
    # test_dataset: testing data to impute
    dist = []
    converted = convert(test_dataset[test_index][0])
    b = (np.sum(converted[indexes[feature_ports]], axis=0))
    a = np.expand_dims(converted[feature_ports], axis=0)
    b = np.expand_dims(b, axis=0)/7
    data = np.concatenate((a,b))
    for i in range(len(train_dataset_knn)):
        d = np.linalg.norm(train_dataset_knn[i][0] - data)
        dist.append({'index': i, 'dist': d})
    dist.sort(key=myFunc)
    x = [i['index'] for i in dist]
    return x[0:k]

def myFunc(e):
    return e['dist']

def prepare_knn_data(train_dataset, indexes):
    train_dataset_knn_even = []
    for i in range(len(train_dataset)):
        converted = convert(train_dataset[i][0])
        for j in range(8):
            b = (np.sum(converted[indexes[j]], axis=0))/7
            a = np.expand_dims(converted[j], axis=0)
            b = np.expand_dims(b, axis=0)
            d = np.concatenate((a,b))
            train_dataset_knn_even.append((d, train_dataset[i][1][2*j]))
    train_dataset_knn_odd = []
    for i in range(len(train_dataset)):
        converted = convert(train_dataset[i][0])
        for j in range(8):
            b = (np.sum(converted[indexes[j]], axis=0))/7
            a = np.expand_dims(converted[j], axis=0)
            b = np.expand_dims(b, axis=0)
            d = np.concatenate((a,b))
            train_dataset_knn_odd.append((d, train_dataset[i][1][2*j+1]))
            
    return train_dataset_knn_even, train_dataset_knn_odd

def knn(config, test_dataset, train_dataset, rackdata_len, ingressBytes_max):
    WINDOW_SKIP = config.window_skip
    WINDOW_SIZE = config.window_size
    COARSE = config.zoom_in_factor
    evenq = np.arange(0,8)
    num_intervals = 33
    skipped = WINDOW_SIZE // WINDOW_SKIP
    indexes = []
    for i in range(8):
        a = list(range(i))
        b = list(range(i+1,8))
        indexes.append((a+b))
    if config.compute_baselines == 'False': 
        # Pre-loaded evaluation data
        with open("evaluation/saved_data/res_pred_knn.pickle", "rb") as fin:
            res_pred_knn = pickle.load(fin)
            fin.close()
        res_true_knn = np.zeros((rackdata_len, 16, 33, 300))
        for q in evenq:
            feature_ports = q
            label_ports_even = q*2
            label_ports_odd = q*2+1
            for i in range(rackdata_len):
                cnt = 0
                for j in range(i*num_intervals, (i+1)*num_intervals):
                    if (j < num_intervals and j % skipped == 0) \
                        or (j >= num_intervals * i and (j - num_intervals * i) % skipped == 0):
                        res_true_knn[i,label_ports_odd,cnt,:] = test_dataset[j][1][label_ports_odd]
                        res_true_knn[i,label_ports_even,cnt,:] = test_dataset[j][1][label_ports_even]
                        cnt += 1
        res_true_knn = np.reshape(res_true_knn, (rackdata_len,16,9900))
    else:
        # Run KNN froms scrach
        k = 4
        res_true_knn = np.zeros((rackdata_len, 16, 33, 300))
        res_pred_knn = np.zeros((rackdata_len, 16, 33, 300))
        train_dataset_knn_even, train_dataset_knn_odd = prepare_knn_data(train_dataset, indexes)
        for q in evenq:
            feature_ports = q
            label_ports_even = q*2
            label_ports_odd = q*2+1
            for i in range(rackdata_len):
                cnt = 0
                for j in range(i*num_intervals, (i+1)*num_intervals):
                    if (j < num_intervals and j % skipped == 0) \
                        or (j >= num_intervals * i and (j - num_intervals * i) % skipped == 0):
                        n = neighbors(j, k, feature_ports, indexes, train_dataset_knn_odd, test_dataset)
                        x = [train_dataset_knn[z][1] for z in n]
                        x = np.array(x)
                        r = np.sum(x, axis = 0) / k
                        res_true_knn[i,label_ports_odd,cnt,:] = test_dataset[j][1][label_ports_odd]
                        res_pred_knn[i,label_ports_odd,cnt,:] = r

                        n = neighbors(j, k, feature_ports, indexes, train_dataset_knn_even, test_dataset)
                        x = [train_dataset_knn[z][1] for z in n]
                        x = np.array(x)
                        r = np.sum(x, axis = 0) / k
                        res_true_knn[i,label_ports_even,cnt,:] = test_dataset[j][1][label_ports_even]
                        res_pred_knn[i,label_ports_even,cnt,:] = r
                        cnt += 1
                        
        res_true_knn = np.reshape(res_true_knn, (rackdata_len,16,9900))
        res_pred_knn = np.reshape(res_pred_knn, (rackdata_len,16,9900))

    res_knn = downstream_task(res_pred_knn, res_true_knn, rackdata_len, ingressBytes_max)

    return res_knn

def plain_transformer(config, test_dataset, rackdata_len, ingressBytes_max):
    WINDOW_SKIP = config.window_skip
    WINDOW_SIZE = config.window_size
    COARSE = config.zoom_in_factor
    model_plain_odd = load_model(config, config.plain_model_dir_odd, d_model=40, n_heads=config.n_heads, dim_feedforward=20, 
                                max_len=36, zoom_in_factor=config.zoom_in_factor, window_size=config.window_size)
    model_plain_odd.eval()
    model_plain_even = load_model(config, config.plain_model_dir_odd, d_model=40, n_heads=config.n_heads, dim_feedforward=20, 
                                max_len=36, zoom_in_factor=config.zoom_in_factor, window_size=config.window_size)
    model_plain_even.eval()

    indexes = []
    for i in range(8):
        a = list(range(i))
        b = list(range(i+1,8))
        indexes.append((a+b))
    even = np.arange(0,8)
    res_true_plain = np.zeros((rackdata_len, 16, 33, 300))
    res_pred_plain = np.zeros((rackdata_len, 16, 33, 300))
    num_intervals = 33
    skipped = WINDOW_SIZE // WINDOW_SKIP
    for q in even:
        feature_ports = q
        label_ports_even = q*2
        label_ports_odd = q*2+1
        for i in range(rackdata_len):
            cnt = 0
            for j in range(i*num_intervals, (i+1)*num_intervals):
                if (j < num_intervals and j % skipped == 0) or \
                (j >= num_intervals * i and (j - num_intervals * i) % skipped == 0):

                    converted = convert_even_plain(test_dataset[j][0])
                    b = (np.sum(converted[indexes[feature_ports]], axis=0))
                    a = np.expand_dims(converted[feature_ports], axis=0)
                    b = np.expand_dims(b, axis=0)/7
                    x = inference_withoutCCA(model_plain_even, np.concatenate((a,b)), COARSE=COARSE)
                    res_true_plain[i,label_ports_even,cnt,:] = test_dataset[j][1][label_ports_even]
                    res_pred_plain[i,label_ports_even,cnt,:] = x[0].cpu().numpy()
                    
                    converted = convert_odd_plain(test_dataset[j][0])
                    b = (np.sum(converted[indexes[feature_ports]], axis=0))
                    a = np.expand_dims(converted[feature_ports], axis=0)
                    b = np.expand_dims(b, axis=0)/7
                    x = inference_withoutCCA(model_plain_odd, np.concatenate((a,b)), COARSE=COARSE)
                    res_true_plain[i,label_ports_odd,cnt,:] = test_dataset[j][1][label_ports_odd]
                    res_pred_plain[i,label_ports_odd,cnt,:] = x[0].cpu().numpy()
                    cnt += 1
    res_true_plain = np.reshape(res_true_plain, (rackdata_len,16,9900))
    res_pred_plain = np.reshape(res_pred_plain, (rackdata_len,16,9900))

    res_plain = downstream_task(res_pred_plain, res_true_plain, rackdata_len, ingressBytes_max)

    return res_plain

def iterimputer_convert(data):
    a = np.zeros((8,6,300))
    a[:,0,:] = np.nan
    a[:,2,:] = np.nan
    for i in range(8):
        for j in range(6):
            a[i,0,j*50] = data[i*2][3][j*50]
            a[i,1,(j+1)*50-1] = data[i*2][0][j*50]
            a[i,2,j*50] = data[i*2+1][3][j*50] 
            a[i,3,(j+1)*50-1] = data[i*2+1][0][j*50] 
            a[i,4,(j+1)*50-1] = data[i*2][1][j*50]
            a[i,5,(j+1)*50-1] = data[i*2][2][j*50]
    return a

def iter_imputer(config, test_dataset, rackdata_len, ingressBytes_max):
    WINDOW_SKIP = config.window_skip
    WINDOW_SIZE = config.window_size
    COARSE = config.zoom_in_factor
    num_intervals = len(np.arange(0,2000,WINDOW_SKIP))-1
    num_WINDOW = len(np.arange(0,2000,WINDOW_SIZE))
    skipped = WINDOW_SIZE // WINDOW_SKIP
    indexes = []
    for i in range(8):
        a = list(range(i))
        b = list(range(i+1,8))
        indexes.append((a+b))
    even = np.arange(0,8)

    if config.compute_baselines == 'False': 
        # Pre-loaded evaluation data
        with open("evaluation/saved_data/res_pred_iter.pickle", "rb") as fin:
            res_pred_iter = pickle.load(fin)
            fin.close()
        res_true_iter = np.zeros((rackdata_len, 16, 33, 300))
        for q in even:
            feature_ports = q
            label_ports_even = q*2
            label_ports_odd = q*2+1
            for i in range(rackdata_len):
                cnt = 0
                for j in range(i*num_intervals, (i+1)*num_intervals):
                    if (j < num_intervals and j % skipped == 0) \
                        or (j >= num_intervals * i and (j - num_intervals * i) % skipped == 0):
                        res_true_iter[i,label_ports_odd,cnt,:] = test_dataset[j][1][label_ports_odd]
                        res_true_iter[i,label_ports_even,cnt,:] = test_dataset[j][1][label_ports_even]
                        cnt += 1
        res_true_iter = np.reshape(res_true_iter, (rackdata_len,16,9900))
    else:
        # Run IterImputer from scrach
        iter_imp = IterativeImputer(random_state=0)
        res_true_iter = np.zeros((rackdata_len, 16, 33, 300))
        res_pred_iter = np.zeros((rackdata_len, 16, 33, 300))
        num_intervals = 33
        skipped = WINDOW_SIZE // WINDOW_SKIP
        for q in even:
            feature_ports = q
            label_ports_even = q*2
            label_ports_odd = q*2+1
            for i in range(rackdata_len):
                cnt = 0
                for j in range(i*num_intervals, (i+1)*num_intervals):
                    if (j < num_intervals and j % skipped == 0) or \
                    (j >= num_intervals * i and (j - num_intervals * i) % skipped == 0):
                        converted = iterimputer_convert(test_dataset[j][0]) # (8,6,300)
                        b = np.sum(converted[indexes[feature_ports]], axis=0)
                        a = converted[feature_ports]
                        b = b/7
                        x = np.concatenate((a,b))
                        res_true_iter[i,label_ports_even,cnt,:] = test_dataset[j][1][label_ports_even]
                        iter_res = iter_imp.fit_transform(x)
                        res_pred_iter[i,label_ports_even,cnt,:] = iter_res[0]
                        
                        res_true_iter[i,label_ports_odd,cnt,:] = test_dataset[j][1][label_ports_odd]
                        res_pred_iter[i,label_ports_odd,cnt,:] = iter_res[2]
                        
                        cnt += 1
        res_true_iter = np.reshape(res_true_iter, (rackdata_len,16,9900))
        res_pred_iter = np.reshape(res_pred_iter, (rackdata_len,16,9900))


    res_iter = downstream_task(res_pred_iter, res_true_iter, rackdata_len, ingressBytes_max)

    return res_iter

def brits(config, test_dataset, train_dataset, rackdata_len, throughput_threshold):
    WINDOW_SKIP = config.window_skip
    WINDOW_SIZE = config.window_size
    COARSE = config.zoom_in_factor
    num_intervals = len(np.arange(0,2000,WINDOW_SKIP))-1
    num_WINDOW = len(np.arange(0,2000,WINDOW_SIZE))
    skipped = WINDOW_SIZE // WINDOW_SKIP
    indexes = []
    for i in range(8):
        a = list(range(i))
        b = list(range(i+1,8))
        indexes.append((a+b))
    even = np.arange(0,8)
    if config.compute_baselines == 'False': 
        # Pre-loaded evaluation data
        with open("evaluation/saved_data/res_pred_brits.pickle", "rb") as fin:
            res_pred_brits = pickle.load(fin)
            fin.close()
        res_true_brits = np.zeros((rackdata_len, 16, 33, 300))
        for q in even:
            feature_ports = q
            label_ports_even = q*2
            label_ports_odd = q*2+1
            for i in range(rackdata_len):
                cnt = 0
                for j in range(i*num_intervals, (i+1)*num_intervals):
                    if (j < num_intervals and j % skipped == 0) \
                        or (j >= num_intervals * i and (j - num_intervals * i) % skipped == 0):
                        res_true_brits[i,label_ports_odd,cnt,:] = test_dataset[j][1][label_ports_odd]
                        res_true_brits[i,label_ports_even,cnt,:] = test_dataset[j][1][label_ports_even]
                        cnt += 1
        res_true_brits = np.reshape(res_true_brits, (rackdata_len,16,9900))
    else:
        # train Brits from scrach
        data_iter_train, data_iter_test = train_brits.prepare_brits_data(config,train_dataset, test_dataset)
        print('start training Brits')
        model = train_brits.train_brits(data_iter_train, data_iter_test, config.window_size)
        res_true_brits, res_pred_brits = train_brits.run_inference(config, test_dataset, model, rackdata_len)
    
    res_brits = downstream_task(res_pred_brits, res_true_brits, rackdata_len, throughput_threshold)

    return res_brits
