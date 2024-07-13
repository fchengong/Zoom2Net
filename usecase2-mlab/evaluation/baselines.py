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


def neighbors(test_index, k, train_dataset, test_dataset):
    # test_index: index into test_dataset
    # k: consider first k closest neighbors 
    # train_dataset: training data
    # test_dataset: testing data to impute
    index = -1
    dist = []
    for i in range(len(train_dataset)):   
        d = np.linalg.norm(train_dataset[i][0] - test_dataset[test_index][0])
        dist.append({'index': i, 'dist': d})
    dist.sort(key=myFunc)
    x = [i['index'] for i in dist]
    return x[0:k]

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

def knn(config, test_dataset, train_dataset, data_3s_train, \
            data_3s_test, data_3to6s_train, data_3to6s_test, data_6to9s_train, data_6to9s_test, \
                index_of_3s, index_of_3to6s, index_of_6to9s):    
    pred9s_knn = []
    true9s_knn = []
    pred6s_knn = []
    true6s_knn = []
    pred3s_knn = []
    true3s_knn = []
    k = 6
    for i in range(data_3s_test):
        n = neighbors(i, k, train_dataset, test_dataset)
        r = [train_dataset[z][1] for z in n]
        r = np.array(r)
        s3 = np.sum(r, axis = 0) / k
        ground_thruth_3s = test_dataset[i][1]    
        a = index_of_3s[data_3s_train+i]
        b = np.where(index_of_3to6s == a)[0]
        c = np.where(index_of_6to9s == a)[0]
        exist_3to6 = False
        exist_6to9 = False
        if len(b) != 0 and b[0] > data_3to6s_train:
            exist_3to6 = True
            test_index = b[0] - data_3to6s_train + data_3s_test
            # print(b[0], test_index, test_index)
            n = neighbors(test_index, k, train_dataset, test_dataset)
            r = [train_dataset[z][1] for z in n]
            r = np.array(r)
            s3to6 = np.sum(r, axis = 0) / k
            ground_thruth_3to6s = test_dataset[i][1]
        if exist_3to6 == True and len(c) != 0 and c[0] > data_6to9s_train:
            exist_6to9 = True
            test_index2 = c[0] - data_6to9s_train + data_3s_test + data_3to6s_test
            # print(c[0], test_index, test_index2)
            n = neighbors(test_index2, k, train_dataset, test_dataset)
            r = [train_dataset[z][1] for z in n]
            r = np.array(r)
            s6to9 = np.sum(r, axis = 0) / k
            ground_thruth_6to9s = test_dataset[i][1]
        if exist_3to6 == True and exist_6to9 == True:
            pred9s_knn.append(np.concatenate((s3, s3to6, s6to9)))
            true9s_knn.append(np.concatenate((ground_thruth_3s, ground_thruth_3to6s, ground_thruth_6to9s)))
        elif exist_3to6 == True and exist_6to9 == False:
            pred6s_knn.append(np.concatenate((s3, s3to6)))
            true6s_knn.append(np.concatenate((ground_thruth_3s, ground_thruth_3to6s)))
        elif exist_3to6 == False and exist_6to9 == False:
            pred3s_knn.append(s3)
            true3s_knn.append(ground_thruth_3s)
    pred9s_knn = np.array(pred9s_knn)
    true9s_knn = np.array(true9s_knn)
    pred6s_knn = np.array(pred6s_knn)
    true6s_knn = np.array(true6s_knn)
    pred3s_knn = np.array(pred3s_knn)
    true3s_knn = np.array(true3s_knn)

    res_knn = downstream_task([true9s_knn, true6s_knn, true3s_knn], [pred9s_knn, pred6s_knn, pred3s_knn])

    return res_knn

def plain_transformer(config, test_dataset, data_3s_train, \
            data_3s_test, data_3to6s_train, data_3to6s_test, data_6to9s_train, data_6to9s_test, \
                index_of_3s, index_of_3to6s, index_of_6to9s):
    model_plain = load_model(config, config.plain_model_dir, d_model=40, n_heads=config.n_heads, dim_feedforward=20)
    model_plain.eval()

    pred9s_plain = []
    true9s_plain = []
    pred6s_plain = []
    true6s_plain = []
    pred3s_plain = []
    true3s_plain = []
    for i in range(data_3s_test):
        s3 = inference(model_plain, (test_dataset[i][0])).cpu().numpy()[0]
        ground_thruth_3s = test_dataset[i][1]
        a = index_of_3s[data_3s_train+i]
        b = np.where(index_of_3to6s == a)[0]
        c = np.where(index_of_6to9s == a)[0]
        exist_3to6 = False
        exist_6to9 = False
        if len(b) != 0 and b[0] > data_3to6s_train:
            exist_3to6 = True
            test_index = b[0] - data_3to6s_train + data_3s_test
            s3to6 = inference(model_plain, (test_dataset[test_index][0])).cpu().numpy()[0]
            ground_thruth_3to6s = test_dataset[test_index][1]
        if exist_3to6 == True and len(c) != 0 and c[0] > data_6to9s_train:
            exist_6to9 = True
            test_index2 = c[0] - data_6to9s_train + data_3s_test + data_3to6s_test
            s6to9 = inference(model_plain, (test_dataset[test_index2][0])).cpu().numpy()[0]
            ground_thruth_6to9s = test_dataset[test_index2][1]
        if exist_3to6 == True and exist_6to9 == True:
            pred9s_plain.append(np.concatenate((s3, s3to6, s6to9)))
            true9s_plain.append(np.concatenate((ground_thruth_3s, ground_thruth_3to6s, ground_thruth_6to9s)))
        elif exist_3to6 == True and exist_6to9 == False:
            pred6s_plain.append(np.concatenate((s3, s3to6)))
            true6s_plain.append(np.concatenate((ground_thruth_3s, ground_thruth_3to6s)))
        elif exist_3to6 == False and exist_6to9 == False:
            pred3s_plain.append(s3)
            true3s_plain.append(ground_thruth_3s)
    pred9s_plain = np.array(pred9s_plain)
    true9s_plain = np.array(true9s_plain)
    pred6s_plain = np.array(pred6s_plain)
    true6s_plain = np.array(true6s_plain)
    pred3s_plain = np.array(pred3s_plain)
    true3s_plain = np.array(true3s_plain)

    res_plain = downstream_task([true9s_plain, true6s_plain, true3s_plain], [pred9s_plain, pred6s_plain, pred3s_plain])

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

def iter_imputer(config, test_dataset, data_3s_train, \
            data_3s_test, data_3to6s_train, data_3to6s_test, data_6to9s_train, data_6to9s_test, \
                index_of_3s, index_of_3to6s, index_of_6to9s):
    if config.compute_baselines == 'False': 
        # Pre-loaded evaluation data
        with open("evaluation/saved_data/res_pred_iter.pickle", "rb") as fin:
            res_pred_iter = pickle.load(fin)
            fin.close()
        with open("evaluation/saved_data/res_true_iter.pickle", "rb") as fin:
            res_true_iter = pickle.load(fin)
            fin.close()
        res_iter = downstream_task(res_true_iter, res_pred_iter)
        return res_iter
    else:
        pred9s_iter = []
        true9s_iter = []
        pred6s_iter = []
        true6s_iter = []
        pred3s_iter = []
        true3s_iter = []
        iter_imp = IterativeImputer(random_state=0)
        for i in range(data_3s_test):
            print(i)
            periodic = np.zeros((1,300))
            periodic[:] = np.nan
            time = [int(a) for a in test_dataset[i][2]]
            periodic[0,time] = test_dataset[i][1][time]
            value = np.concatenate((periodic, test_dataset[i][0]))
            iter_res = iter_imp.fit_transform(value)
            s3 = iter_res[0]
            ground_thruth_3s = test_dataset[i][1]
            a = index_of_3s[data_3s_train+i]
            b = np.where(index_of_3to6s == a)[0]
            c = np.where(index_of_6to9s == a)[0]
            exist_3to6 = False
            exist_6to9 = False
            if len(b) != 0 and b[0] > data_3to6s_train:
                exist_3to6 = True
                test_index = b[0] - data_3to6s_train + data_3s_test
                periodic = np.zeros((1,300))
                periodic[:] = np.nan
                time = [int(a) for a in test_dataset[test_index][2]]
                periodic[0,time] = test_dataset[test_index][1][time]
                value = np.concatenate((periodic, test_dataset[test_index][0]))
                iter_res = iter_imp.fit_transform(value)
                s3to6 = iter_res[0]
                ground_thruth_3to6s = test_dataset[test_index][1]                
            if exist_3to6 == True and len(c) != 0 and c[0] > data_6to9s_train:
                exist_6to9 = True
                test_index2 = c[0] - data_6to9s_train + data_3s_test + data_3to6s_test
                periodic = np.zeros((1,300))
                periodic[:] = np.nan
                time = [int(a) for a in test_dataset[test_index2][2]]
                periodic[0,time] = test_dataset[test_index2][1][time]
                value = np.concatenate((periodic, test_dataset[test_index2][0]))
                iter_res = iter_imp.fit_transform(value)
                s6to9 = iter_res[0]
                ground_thruth_6to9s = test_dataset[test_index2][1]                
            if exist_3to6 == True and exist_6to9 == True:
                pred9s_iter.append(np.concatenate((s3, s3to6, s6to9)))
                true9s_iter.append(np.concatenate((ground_thruth_3s, ground_thruth_3to6s, ground_thruth_6to9s)))
            elif exist_3to6 == True and exist_6to9 == False:
                pred6s_iter.append(np.concatenate((s3, s3to6)))
                true6s_iter.append(np.concatenate((ground_thruth_3s, ground_thruth_3to6s)))
            elif exist_3to6 == False and exist_6to9 == False:
                pred3s_iter.append(s3)
                true3s_iter.append(ground_thruth_3s)
        pred9s_iter = np.array(pred9s_iter)
        pred6s_iter = np.array(pred6s_iter)
        pred3s_iter = np.array(pred3s_iter)


    res_iter = downstream_task([true9s_iter, true6s_iter, true3s_iter], [pred9s_iter, pred6s_iter, pred3s_iter])

    return res_iter

def brits(config, test_dataset, train_dataset, data_3s_train, \
                data_3s_test, data_3to6s_train, data_3to6s_test, data_6to9s_train, data_6to9s_test, \
                index_of_3s, index_of_3to6s, index_of_6to9s):
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
        res_true_brits, res_pred_brits = train_brits.run_inference(config, test_dataset, model, data_3s_train, \
                data_3s_test, data_3to6s_train, data_3to6s_test, data_6to9s_train, data_6to9s_test, \
                index_of_3s, index_of_3to6s, index_of_6to9s)
    
    res_brits = downstream_task(res_pred_brits, res_true_brits)

    return res_brits
