import math
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
import pickle

# Project modules
from evaluation.downstream_task import downstream_task
from evaluation.run_inference import impute_data, load_model
from evaluation.baselines import knn, plain_transformer, iter_imputer, brits
from model_training.transformer import TSTransformerEncoder

def run_downstream_task(config, test_dataset, train_dataset, maxsentBytes,\
            data_3s_train, data_3s_test, data_3to6s_train, data_3to6s_test, data_6to9s_train, data_6to9s_test):
    with open("./datasets/mlab_data/index_of_first9s.pickle", "rb") as fin:
        index_of_first9s= pickle.load(fin)
        fin.close()
    index_of_3s = index_of_first9s[0]
    index_of_3to6s = index_of_first9s[1]
    index_of_6to9s = index_of_first9s[2]
    index_of_3s = np.array(index_of_3s)
    index_of_3to6s = np.array(index_of_3to6s)
    index_of_6to9s = np.array(index_of_6to9s)
    timing = False

    res_brits = brits(config, test_dataset, train_dataset, data_3s_train, \
                data_3s_test, data_3to6s_train, data_3to6s_test, data_6to9s_train, data_6to9s_test, \
                index_of_3s, index_of_3to6s, index_of_6to9s)
    # print('brits')

    model = load_model(config, config.z2n_model_dir, d_model=40, n_heads=config.n_heads, dim_feedforward=20)
    model.eval()
    res_pred_z2n, res_true_z2n = impute_data(config, model, test_dataset, timing, maxsentBytes, data_3s_train, \
                data_3s_test, data_3to6s_train, data_3to6s_test, data_6to9s_train, data_6to9s_test, \
                index_of_3s, index_of_3to6s, index_of_6to9s)
    res_z2n = downstream_task(res_pred_z2n, res_true_z2n)
    print('z2n')

    res_knn = knn(config, test_dataset, train_dataset, data_3s_train, \
            data_3s_test, data_3to6s_train, data_3to6s_test, data_6to9s_train, data_6to9s_test, \
                index_of_3s, index_of_3to6s, index_of_6to9s)
    print('knn')

    res_plain = plain_transformer(config, test_dataset, data_3s_train, \
            data_3s_test, data_3to6s_train, data_3to6s_test, data_6to9s_train, data_6to9s_test, \
                index_of_3s, index_of_3to6s, index_of_6to9s)
    print('plain')

    res_iter = iter_imputer(config, test_dataset, data_3s_train, \
            data_3s_test, data_3to6s_train, data_3to6s_test, data_6to9s_train, data_6to9s_test, \
                index_of_3s, index_of_3to6s, index_of_6to9s)
    print('iter')
    plot(res_z2n, res_knn, res_iter, res_plain, res_brits)
        
def plot(res_z2n, res_knn, res_iter, res_plain, res_brits):
  
    a = []
    b = []
    c = []
    d = []
    e = []
    for n,i in enumerate(res_plain.keys()):
        if n == 4:
            break
        a.append(res_z2n[i])
        b.append(res_knn[i])
        c.append(res_iter[i])
        d.append(res_plain[i])
        e.append(res_brits[i])
        
    all_methods = [a,b,c,d,e]

    maxes = np.zeros(4)
    for i in range(len(all_methods)):
        for j in range(4):
            if max(all_methods[i][j]) > maxes[j]:
                maxes[j] = max(all_methods[i][j])
    means = np.zeros(4)
    for i in range(len(all_methods)):
        for j in range(4):
            if np.average(all_methods[i][j]) > means[j]:
                means[j] = np.average(all_methods[i][j])

    stats = ['MSE', 'EMD', 'Autocorrelation', '99percentile']
    methods = ['Zoom2Net', 'KNN', 'CoarseGrained', 'PlainTransformer', 'Brits']
    fig, ax = plt.subplots(1, 1, figsize=(6,3))
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True) 
    cmap = plt.colormaps.get_cmap('Blues')
    a = None
    for i in range(5):
        r = all_methods[i].copy()
        for j in range(4):
            # if j in [0,2,3,4]:
                # r[j] /= maxes[j]
            r[j] /= means[j] * 1.1
    #     x = np.arange(3)
        x = np.array([0,2,4,6])
        width = 0.3
        mean = np.mean(r,axis=1)
        err_lo = mean - np.min(r,axis=1)
        err_hi = np.max(r,axis=1) - mean
        above_threshold = 0
        below_threshold = mean
        print(methods[i], mean)
        ax.bar((x+(i-2)* width), below_threshold, width, label = methods[i],\
            error_kw=dict(lw=1, capsize=1, capthick=1),capsize=2, color=cmap(i*50), edgecolor='k')
    ax.set_xticks(x)
    ax.set_xticklabels(stats, fontsize=11)
    ax.legend(fontsize=11, ncol=3, loc='upper left')
    ax.grid(linestyle='--', axis='y')
    ax.set_axisbelow(True)

    a = []
    b = []
    c = []
    d = []
    e = []
    stats = []
    for n,i in enumerate(res_plain.keys()):
        if n < 4:
            continue
        if n > 13:
            break
        a.append(res_z2n[i])
        b.append(res_knn[i])
        c.append(res_iter[i])
        d.append(res_plain[i])
        e.append(res_brits[i])
        stats.append(i)
    num_web = len(a)
    all_methods = [a,b,c,d,e]

    maxes = np.zeros(len(a))
    for i in range(len(all_methods)):
        for j in range(len(a)):
            if (all_methods[i][j]) > maxes[j]:
                maxes[j] = (all_methods[i][j])
    
    methods = ['Zoom2Net', 'KNN', 'CoarseGrained', 'PlainTransformer', 'Brits']
    fig, ax = plt.subplots(1, 1, figsize=(8,4))
    cmap = plt.colormaps.get_cmap('Blues')
    b = None
    diff = 0
    for i in range(5):
        r = all_methods[i].copy()
        for j in range(num_web):
            r[j] /= maxes[j]*1.1
        x = np.arange(0,30,3)
        width = 0.45
        print(methods[i], r)
            
        if i == 0:
            b = np.array(r)
        else:
            diff += sum(np.array(r) - b)/num_web
        ax.bar(x+(i-1.5)* width, r, width, label = methods[i],\
            capsize=2, color=cmap(i*50), edgecolor='k')
    ax.set_xticks(x+0.25)
    ax.set_xticklabels(stats, rotation=30)
    ax.legend(fontsize=13, ncol=3, loc='upper left')
    ax.grid(linestyle='--', axis='y')
    ax.set_axisbelow(True)
    # ax.set_xlabel('Websites')
    # ax.set_ylabel('Error of webpage \nloading time estimation')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)