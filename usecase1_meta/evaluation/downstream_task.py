import numpy as np
from sklearn.metrics import mean_squared_error
from fastdtw import fastdtw
import ot
line_rate = 12.5*1.25e8/1000


def downstream_task(res_pred, res_true, rackdata_len, ingressBytes_max):
    result = {}
    result['MSE'] = []
    result['emd'] = []
    result['Autocorrelation'] = []
    result['Total_ingress'] = []
    result['Burst_start_pos'] = []
    result['Burst_height'] = []
    result['Burst_freq'] = []
    result['Burst_duration'] = []
    result['Burst_volume'] = []
    result['IngressAfterBurst'] = []
    result['emd'] = []
    result['99_percentile'] = []
    mse = []
    acorr = []
    ingressMax = []
    num_bursts = []
    bursts_start_pos = []
    burstduration = []
    burst_volume = []
    ingressAfterBurst = []
    totalingress= []
    emd = []
    p99 = []


    normalization = ingressBytes_max
    for i in range(rackdata_len*92):
        
        bursts_val_true, bursts_index_true = burst_detection(res_true[i], line_rate/2/normalization)
        # print('get here1')
        # if len(bursts_val_true) == 0:
        #     continue
        bursts_val_pred, bursts_index_pred = burst_detection(res_pred[i], line_rate/2/normalization)
        
    ####################  MSE  ##################
        # print('get here2')
        mse.append(mean_squared_error(res_true[i], res_pred[i]))
    
    ###################  Auto correlation  #############
        a1 = autocorr(res_true[i])
        a2 = autocorr(res_pred[i])
        acorr.append(mean_squared_error(a1,a2))

    ####################  EMD  ##################
        emd.append(ot.wasserstein_1d(res_true[i], res_pred[i]))
    
    ####################  P99  ##################
        p99_true = np.percentile(res_true[i], 99)
        p99_pred = np.percentile(res_pred[i], 99)
        if p99_true == 0:
            p99.append(abs(p99_true - p99_pred) / 1)
        else:
            p99.append(abs(p99_true - p99_pred) / p99_true)
    
    ####################  Num bursts/Burst freq  ##################
        num_bursts_pred = len(bursts_val_pred)
        num_bursts_true = len(bursts_val_true)
        if num_bursts_true == 0:
            num_bursts.append(abs(num_bursts_true - num_bursts_pred) / 1)
        else:
            if num_bursts_pred == 0:
                num_bursts.append(abs(num_bursts_true - num_bursts_pred) / 1)
            else:
                num_bursts.append(abs(num_bursts_true - num_bursts_pred) / num_bursts_true)
    
    ####################  Burst height  ##################
        ingressMax_true = []
        ingressMax_pred = []
        for j in bursts_val_true:
            ingressMax_true.append(max(j))
        for j in bursts_val_pred:
            ingressMax_pred.append(max(j))
        
        ingressMax_true = sum(ingressMax_true)/len(ingressMax_true) if len(ingressMax_true)!=0 else 0
        ingressMax_pred = sum(ingressMax_pred)/len(ingressMax_pred) if len(ingressMax_pred)!=0 else 0
        if ingressMax_true == 0:
            ingressMax.append(abs(ingressMax_true - ingressMax_pred) / 1)
        else:
            ingressMax.append(abs(ingressMax_true - ingressMax_pred) / ingressMax_true)
    
    ####################  Burst start position  ##################
        bursts_start_pos_pred = []
        bursts_start_pos_true = []
        # for j in bursts_index_true:
        #     bursts_start_pos_true.append(j[0])
        # for j in bursts_index_pred:
        #     bursts_start_pos_pred.append(j[0])
        for n, j in enumerate(bursts_val_true):
            bursts_start_pos_true.append(np.argmax(j)+bursts_index_true[n][0])
        for n, j in enumerate(bursts_val_pred):
            bursts_start_pos_pred.append(np.argmax(j)+bursts_index_pred[n][0])
        if len(bursts_start_pos_pred) == 0:
            # bursts_start_pos_pred = [0]
            bursts_start_pos_pred = [len(res_true[i])]
            # print(len(bursts_start_pos_true))
        distance, path = fastdtw(bursts_start_pos_true, bursts_start_pos_pred)
        bursts_start_pos.append(distance)
    
    ####################  Burst duration  ##################
        burstduration_true = []
        burstduration_pred = []
        for j in bursts_index_true:
            burstduration_true.append(len(j))
        for j in bursts_index_pred:
            burstduration_pred.append(len(j))
        
        burstduration_true = sum(burstduration_true) / len(burstduration_true) \
                                            if len(burstduration_true) != 0 else 0
        burstduration_pred = sum(burstduration_pred) / len(burstduration_pred) \
                                            if len(burstduration_pred) != 0 else 0
        if burstduration_true == 0:
            burstduration.append(abs(burstduration_true - burstduration_pred) / 1)
        else:
            burstduration.append(abs(burstduration_true - burstduration_pred) / burstduration_true)
            
    ####################  Burst volume  ##################
        burstvol_true = []
        burstvol_pred = []
        for j in bursts_val_true:
            burstvol_true.append(sum(j))
        for j in bursts_val_pred:
            burstvol_pred.append(sum(j))
        burstvol_true = sum(burstvol_true)/len(burstvol_true) if len(burstvol_true) != 0 else 0
        burstvol_pred = sum(burstvol_pred)/len(burstvol_pred) if len(burstvol_pred) != 0 else 0
        if burstvol_true == 0:
            burst_volume.append(abs(burstvol_true - burstvol_pred) / 1)
        else:    
            burst_volume.append(abs(burstvol_true - burstvol_pred) / burstvol_true)
    
    ####################  After Burst volume  ##################
        ingressAfterBurst_true = []
        ingressAfterBurst_pred = []
        for j in bursts_index_true:
            if j[-1]+1 >= len(res_true[i]):
                continue
            ingressAfterBurst_true.append(res_true[i][j[-1]+1])
        for j in bursts_index_pred:
            if j[-1]+1 >= len(res_pred[i]):
                continue
            ingressAfterBurst_pred.append(res_pred[i][j[-1]+1])
        # print(ingressAfterBurst_true)
        # print(ingressAfterBurst_pred)
        ingressAfterBurst_true = sum(ingressAfterBurst_true)/len(ingressAfterBurst_true) if len(ingressAfterBurst_true) != 0 else 0
        ingressAfterBurst_pred = sum(ingressAfterBurst_pred)/len(ingressAfterBurst_pred) if len(ingressAfterBurst_pred) != 0 else 0
    
        if ingressAfterBurst_true == 0:
            e = abs(ingressAfterBurst_true - ingressAfterBurst_pred) / 1
        else:    
            # if ingressAfterBurst_pred == 0:
            e = abs(ingressAfterBurst_true - ingressAfterBurst_pred) / ingressAfterBurst_true
        # print(e)
        # print('===================')
        ingressAfterBurst.append(e)
            
    ####################  total ingress  ##################
        totalingress_true = sum(res_true[i])
        totalingress_pred = sum(res_pred[i])
        
        if totalingress_true == 0:
            totalingress.append(abs(totalingress_true - totalingress_pred) / 1)
        else:
            totalingress.append(abs(totalingress_true - totalingress_pred) / totalingress_true)

    mse_final = np.round(sum(mse) / len(mse),5)
    acorr_final = np.round(sum(acorr) / len(acorr),5)
    emd_final = np.round(sum(emd) / len(emd),5)
    p99_final = np.round(sum(p99) / len(p99),5)
    ingressMax_final = np.round(sum(ingressMax) / len(ingressMax),5)
    num_bursts_final = np.round(sum(num_bursts) / len(num_bursts),5)
    bursts_start_pos_final = np.round(sum(bursts_start_pos) / len(bursts_start_pos),5)
    burstduration_final = np.round(sum(burstduration) / len(burstduration),5)
    burst_volume_final = np.round(sum(burst_volume) / len(burst_volume),5)
    ingressAfterBurst_final = np.round(sum(ingressAfterBurst) / len(ingressAfterBurst),5)
    totalingress_final= np.round(sum(totalingress) / len(totalingress),5)

    result['MSE'].append(mse_final)
    result['emd'].append(emd_final)
    result['99_percentile'].append(p99_final)
    result['Autocorrelation'].append(acorr_final)
    result['Total_ingress'].append(totalingress_final)
    result['Burst_start_pos'].append(bursts_start_pos_final)
    result['Burst_height'].append(ingressMax_final)
    result['Burst_freq'].append(num_bursts_final)
    result['Burst_duration'].append(burstduration_final)
    result['Burst_volume'].append(burst_volume_final)
    result['IngressAfterBurst'].append(ingressAfterBurst_final)
    # result['MSE'] = (mse_final)
    # result['emd'] = (emd_final)
    # result['99_percentile'] = (p99_final)
    # result['Autocorrelation'] = (acorr_final)
    # result['Total_ingress'] = (totalingress_final)
    # result['Burst_start_pos'] = (bursts_start_pos_final)
    # result['Burst_height'] = (ingressMax_final)
    # result['Burst_freq'] = (num_bursts_final)
    # result['Burst_duration'] = (burstduration_final)
    # result['Burst_volume'] = (burst_volume_final)
    # result['IngressAfterBurst'] = (ingressAfterBurst_final)

    return result

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]

def burst_detection(data, threshold):
    res = []
    bursts_val = []
    bursts_index = []
    peak_pos = []
    height = []
    burst = False
    for i in range(len(data)):
        temp = []
        temp_ind = []
        if burst == False:
            if data[i] > threshold:
                burst = True
                temp.append(data[i])
                temp_ind.append(i)
                j = i
                while j + 1 < len(data) - 1:
                    j = j + 1
                    if data[j] >= threshold:
                        temp.append(data[j])
                        temp_ind.append(j)
                    else:
                        break
                bursts_val.append(temp)
                bursts_index.append(temp_ind)
        else:
            if data[i] < threshold:
                burst = False
    return bursts_val, bursts_index
