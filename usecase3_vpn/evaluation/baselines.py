import numpy as np
import itertools
import torch
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from evaluation.downstream_task import downstream_task
from model_training.transformer import TSTransformerEncoder
from model_training.utils import inference
from evaluation.run_inference import load_model


    # test_index: index into test_dataset
    # feature_ports: which queue to be imputed
    # k: consider first k closest neighbors 
    # train_dataset_knn: training data
    # test_dataset: testing data to impute
def near_in_x(i, X_test, X_train):
    dist = []
    label = 0
    for j in range(len(X_train)):
        if i == j:
            continue
        d = np.linalg.norm(X_train[j] - X_test[i])
        dist.append({'index': j, 'dist': d})
    dist.sort(key=myFunc)
    return dist

def myFunc(e):
    return e['dist']

def prepare_knn_data(train_dataset, indexes):
    train_dataset_knn = []
    for i in range(len(train_dataset)):
        converted = convert(train_dataset[i][0])
        for j in range(8):
            b = (np.sum(converted[indexes[j]], axis=0))/7
            a = np.expand_dims(converted[j], axis=0)
            b = np.expand_dims(b, axis=0)
            d = np.concatenate((a,b))
            train_dataset_knn.append((d, train_dataset[i][1][2*j+1]))
            
    return train_dataset_knn

def knn(config, X_train, Y_train, X_test, Y_test):
    WINDOW_SKIP = config.window_skip
    WINDOW_SIZE = config.window_size
    COARSE = config.zoom_in_factor
    evenq = np.arange(0,8)
    num_intervals = 33
    skipped = WINDOW_SIZE // WINDOW_SKIP
    indexes = []
    res_pred_knn = []
    res_true_knn = []
    for i in range(len(X_test)):
        dist = near_in_x(i, X_test, X_train)
        r1 = np.zeros(40)
        for k in dist[0:10]:
            r1 += Y_train[k['index']][0]
        r1 = r1/10
        r2 = np.zeros(40)
        for k in dist[0:10]:
            r2 += Y_train[k['index']][1]
        r2 = r2/10
        r3 = np.zeros(40)
        for k in dist[0:10]:
            r3 += Y_train[k['index']][2]
        r3 = r3/10
        r4 = np.zeros(40)
        for k in dist[0:10]:
            r4 += Y_train[k['index']][3]
        r4 = r4/10
        res_pred_knn.append(np.concatenate((r1,r2,r3,r4)))
        res_true_knn.append(Y_test[i].flatten())
    res_knn = downstream_task(res_pred_knn, res_true_knn, True)

    return res_knn

def plain_transformer(config, X_test, Y_test):
    WINDOW_SKIP = config.window_skip
    WINDOW_SIZE = config.window_size
    COARSE = config.zoom_in_factor
    model_plain = load_model(config, config.plain_model_dir)
    model_plain.eval()

    res_true_plain = []
    res_pred_plain = []
    for i in range(len(X_test)):
        x = inference(model_plain, X_test[i])[0].cpu().numpy()
        x = x[0].reshape((4,40))
        # x = post_process_iat(X_test[i], X_test[i][3]*(maximum_deleted[3] - minimum_deleted[3])+minimum_deleted[3], x, log=False)
        # x = post_process_pktlen(X_test[i], x, log=False)
        res_pred_plain.append(np.concatenate((x[0], x[1], x[2], x[3])))
        res_true_plain.append(np.concatenate((Y_test[i][0], Y_test[i][1], Y_test[i][2], Y_test[i][3])))

    res_plain = downstream_task(res_pred_plain, res_true_plain, False)

    return res_plain

def near_in_x_withinput(i, input):
    dist = []
    def myFunc(e):
      return e['dist']
    label = 0
    for j in range(len(input)):
        if i == j:
            continue
        d = np.linalg.norm(input[j] - input[i])
        dist.append({'index': j, 'dist': d})
    dist.sort(key=myFunc)
    return dist

def knn_new_features(input, output, y1_min, y1_max, y2_min, y2_max, flat_res):
    fiat_total_knn = []
    biat_total_knn = []
    for i in range(len(input)):
        dist = near_in_x_withinput(i, input)
        r1 = np.zeros(40)
        for k in dist[0:10]:
            r1 += output[k['index']][1]
        r1 = r1/10
        total = sum(r1)* (y2_max - y2_min)
        fiat_total_knn.append(total)
        r1 = np.zeros(40)
        for k in dist[0:10]:
            r1 += output[k['index']][3]
        r1 = r1/10
        total = sum(r1)* (y2_max - y2_min)
        biat_total_knn.append(total)
    fiat_mean_knn = []
    biat_mean_knn = []
    duration_knn = []
    for i in range(len(flat_res)):
        fiat_mean_knn.append(fiat_total_knn[i] / (flat_res[i][0][2]-1))
        biat_mean_knn.append(biat_total_knn[i] / (flat_res[i][0][3]-1))
        if fiat_total_knn[i] > biat_total_knn[i]:
            duration_knn.append(fiat_total_knn[i])
        else:
            duration_knn.append(biat_total_knn[i])
    fb_psec_knn = []
    for i in range(len(input)):
        dist = near_in_x_withinput(i, input)
        r1 = np.zeros(40)
        for k in dist[0:10]:
            r1 += output[k['index']][0]
        r1 = r1/10
        r2 = np.zeros(40)
        for k in dist[0:10]:
            r2 += output[k['index']][2]
        r2 = r2/10
        total = sum(np.concatenate((r1,r2))) * (y1_max - y1_min)
        duration = duration_knn[i] 
        fb_psec_knn.append(total / (duration / (10**6)))
    return fiat_total_knn, biat_total_knn, fiat_mean_knn, biat_mean_knn, duration_knn, fb_psec_knn

def plain_new_features(model, input, output, y1_min, y1_max, y2_min, y2_max):
    fiat_total_plain = []
    biat_total_plain = []
    for i in range(len(input)):
        x = inference(model, input[i])[0].cpu().numpy()
        x = x[0].reshape((4,40))
        # x = post_process_iat(input[i], flat_res[i][0][4], x, log=False)
        fiat_total_plain.append(sum(x[1])* (y2_max - y2_min))
        biat_total_plain.append(sum(x[3])* (y2_max - y2_min))
    fiat_mean_plain = []
    biat_mean_plain = []
    duration_plain = []
    for i in range(len(fiat_total_plain)):
        fiat_mean_plain.append(fiat_total_plain[i] / (flat_res[i][0][2]-1))
        biat_mean_plain.append(biat_total_plain[i] / (flat_res[i][0][3]-1))
        if fiat_total_plain[i] > biat_total_plain[i]:
            duration_plain.append(fiat_total_plain[i])
        else:
            duration_plain.append(biat_total_plain[i])
    fb_psec_plain = []
    for i in range(len(input)):
        x = inference(model, input[i])[0].cpu().numpy()
        x = x[0].reshape((4,40))
        
        total = sum(np.concatenate((x[0], x[2]))) * (y1_max - y1_min)
        duration = duration_plain[i] 
        fb_psec_plain.append(total / (duration / (10**6)))
        
    return fiat_total_plain, biat_total_plain, fiat_mean_plain, biat_mean_plain, duration_plain, fb_psec_plain