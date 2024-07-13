import math
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
import pickle

# Project modules
from evaluation.downstream_task import downstream_task
from evaluation.run_inference import impute_data, load_model, z2n_new_features
from evaluation.baselines import knn, plain_transformer, knn_new_features, plain_new_features
from evaluation.classifier_training import train_classifier
from model_training.transformer import TSTransformerEncoder

def run_downstream_task(config, X_train, Y_train, X_test, Y_test, maximum_deleted, minimum_deleted,\
                y1_min, y1_max, y2_min, y2_max):
    timing = False
    model = load_model(config, config.z2n_model_dir)
    model.eval()
    res_pred_z2n, res_true_z2n = impute_data(config, model, X_test, Y_test, maximum_deleted, \
                    minimum_deleted, y1_min, y1_max, y2_min, y2_max)
    res_z2n = downstream_task(res_pred_z2n, res_true_z2n, False)
    print('z2n')

    res_knn = knn(config, X_train, Y_train, X_test, Y_test)
    print('knn')

    res_plain = plain_transformer(config, X_test, Y_test)
    print('plain')

    plot(res_z2n, res_knn, res_plain)

def run_new_features(config, input, output, y1_min, y1_max, y2_min, y2_max, maximum_deleted, minimum_deleted):
    with open("./datasets/vpn_data/len40_feat17.pickle", "rb") as fin:
        flat_res= pickle.load(fin)
        fin.close()
    # KNN
    # fiat_total_knn, biat_total_knn, fiat_mean_knn, biat_mean_knn, duration_knn, fb_psec_knn=\
    #     knn_new_features(input, output, y1_min, y1_max, y2_min, y2_max, flat_res)
    # train_classifier(fiat_total_knn, biat_total_knn, fiat_mean_knn, biat_mean_knn, duration_knn, fb_psec_knn)

    # Z2N
    model = load_model(config, config.z2n_model_dir)
    model.eval()
    fiat_total_z2n, biat_total_z2n, fiat_mean_z2n, biat_mean_z2n, duration_z2n, fb_psec_z2n=\
        z2n_new_features(model, input, output, y1_min, y1_max, y2_min, y2_max, maximum_deleted, minimum_deleted, flat_res)
    train_classifier(fiat_total_z2n, biat_total_z2n, fiat_mean_z2n, biat_mean_z2n, duration_z2n, fb_psec_z2n)

    fiat_total_plain, biat_total_plain, fiat_mean_plain, biat_mean_plain, duration_plain, fb_psec_plain=\
                                plain_new_features(model, input, output, y1_min, y1_max, y2_min, y2_max, flat_res)
    train_classifier(fiat_total_plain, biat_total_plain, fiat_mean_plain, biat_mean_plain, duration_plain, fb_psec_plain)

def plot(res_z2n, res_knn, res_plain):
    a = np.stack((res_z2n['MSE'], res_z2n['emd'], res_z2n['Autocorrelation'], res_z2n['99_percentile'])) # (3, 10)
    b = np.stack((res_plain['MSE'], res_plain['emd'], res_plain['Autocorrelation'], res_plain['99_percentile']))
    c = np.stack((res_knn['MSE'], res_knn['emd'], res_knn['Autocorrelation'], res_knn['99_percentile']))

    all_methods = [a,b,c]
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
    methods = ['Zoom2Net', 'PlainTransformer', 'KNN']
    fig, ax = plt.subplots(1, 1, figsize=(6,3))
    cmap = plt.colormaps.get_cmap('Blues')
    a = None
    for i in range(3):
        r = all_methods[i].copy()
        for j in range(4):
            # if j in [0,2,3,4]:
                # r[j] /= maxes[j]
            r[j] /= means[j] * 1.1
    #     x = np.arange(3)
        x = np.array([1,2,3,4])
        width = 0.2
        mean = np.mean(r,axis=1)
        err_lo = mean - np.min(r,axis=1)
        err_hi = np.max(r,axis=1) - mean
        above_threshold = 0
        below_threshold = mean
        ax.bar((x+(i-1)* width), below_threshold, width, label = methods[i],\
            error_kw=dict(lw=1, capsize=1, capthick=1),capsize=2, color=cmap(i*50), edgecolor='k')
    ax.set_xticks(x)
    ax.set_xticklabels(stats, fontsize=11)
    ax.legend(fontsize=11, ncol=3, loc='upper left')
    ax.grid(linestyle='--', axis='y')
    ax.set_axisbelow(True)
