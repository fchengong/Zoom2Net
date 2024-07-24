import numpy as np
from sklearn.metrics import mean_squared_error
from fastdtw import fastdtw
import ot
line_rate = 12.5*1.25e8/1000


def downstream_task(res_pred, res_true, rackdata_len, throughput_threshold):
    result = {}
    result['MSE'] = []
    result['Burst_start_pos'] = []
    result['Burst_height'] = []
    result['Burst_freq'] = []
    result['Burst_interarrival'] = []
    result['Burst_duration'] = []
    result['Burst_volume'] = []
    result['Empty_queues'] = []
    result['90_percentile'] = []
    result['99_percentile'] = []
    result['emd'] = []
    result['Autocorrelation'] = []
    label_ports = np.arange(1,16,2)
    num_queue = 8

    for a in range(rackdata_len//2):
        mse_perqueue = 0
        acorr_perqueue = 0
        zero_perqueue = 0
        p90_perqueue = 0
        p99_perqueue = 0
        num_bursts_perqueue = 0
        burstmax_perqueue = 0
        burststart_perqueue = 0
        burstduration_perqueue = 0
        burstvol_perqueue = 0
        burst_interarrival_perqueue = 0
        emd_perqueue = 0

        for queue in range(8):
            res_true = np.array(res_true)
            res_pred = np.array(res_pred)
            mse = []
            acorr = []
            zero = []
            p90 = []
            p99 = []
            num_bursts = []
            burstmax = []
            burststart = []
            burstduration = []
            burstvol = []
            burst_interarrival = []
            emd = []
            for setting in range(a*2,a*2+2):
                mse.append(mean_squared_error(res_true[setting][queue], res_pred[setting][queue]))
                
                emd.append(ot.wasserstein_1d(res_true[setting][queue], res_pred[setting][queue]))

                a1 = autocorr(res_true[setting][queue])
                a2 = autocorr(res_pred[setting][queue])
                acorr.append(mean_squared_error(a1,a2))


                zero_true = np.count_nonzero(res_true[setting][queue]==0)
                zero_pred = np.count_nonzero(res_pred[setting][queue]==0)
                zero.append(abs(zero_true - zero_pred) / zero_true)

                p90_true = np.percentile(res_true[setting][queue], 90)
                p90_pred = np.percentile(res_pred[setting][queue], 90)
                if p90_true == 0:
                    p90.append(abs(p90_true - p90_pred) / 1)
                else:
                    p90.append(abs(p90_true - p90_pred) / p90_true)
                    
                p99_true = np.percentile(res_true[setting][queue], 99)
                p99_pred = np.percentile(res_pred[setting][queue], 99)
                if p99_true == 0:
                    p99.append(abs(p99_true - p99_pred) / 1)
                else:
                    p99.append(abs(p99_true - p99_pred) / p99_true)

                th = np.amax(throughput_threshold[setting][label_ports[queue]])
#                 th = np.amax(d_test[setting].throughput_data[64+label_ports[queue]]) / maximum_throughput
                q_max_01 = 0.1
                q_max_03 = 0.3
                bursts_val_true, bursts_index_true = burst_detection(res_true[setting][queue], th, q_max_01, q_max_03)
                bursts_val_pred, bursts_index_pred = burst_detection(res_pred[setting][queue], th, q_max_01, q_max_03)
                if len(bursts_val_true) == 0:
                    continue
                for _ in range(len(res_true)):
                    stop = True
                    for i in range(len(bursts_index_true)):
                        if len(bursts_index_true[i]) > 96:
                            bursts_index_true.pop(i)
                            bursts_val_true.pop(i)
                            stop = False
                            break
                    if stop:
                        break
                for _ in range(len(res_true)):
                    stop = True
                    for i in range(len(bursts_index_pred)):
                        if len(bursts_index_pred[i]) > 96:
                            bursts_index_pred.pop(i)
                            bursts_val_pred.pop(i)
                            stop = False
                            break
                    if stop:
                        break

#                 print(bursts_val_true, bursts_index_true)
#                 print('============')
#                 print(bursts_val_pred, bursts_index_pred)
                num_burst_true = len(bursts_val_true)
                num_burst_pred = len(bursts_val_pred)
    #             print('num_burst_true: ', num_burst_true, 'num_burst_pred: ', num_burst_pred)
                if num_burst_true == 0:
                    num_bursts.append(abs(num_burst_true - num_burst_pred) / 1)
                else:
                    num_bursts.append(abs(num_burst_true - num_burst_pred) / num_burst_true)

                burstmax_true = []
                burstmax_pred = []
                for i in bursts_val_true:
                    burstmax_true.append(max(i))
                for i in bursts_val_pred:
                    burstmax_pred.append(max(i))
                burstmax_true = sum(burstmax_true)/len(burstmax_true) if len(burstmax_true)!=0 else 0
                burstmax_pred = sum(burstmax_pred)/len(burstmax_pred) if len(burstmax_pred)!=0 else 0
                if burstmax_true == 0:
                    burstmax.append(abs(burstmax_true - burstmax_pred) / 1)
                else:
                    burstmax.append(abs(burstmax_true - burstmax_pred) / burstmax_true)

                burststart_true = []
                burststart_pred = []
                for n, i in enumerate(bursts_val_true):
#                     burststart_true.append(i[0])
                    burststart_true.append(np.argmax(i)+bursts_index_true[n][0])
                for n, i in enumerate(bursts_val_pred):
                    burststart_pred.append(np.argmax(i)+bursts_index_pred[n][0])
                if len(burststart_pred) == 0:
                    burststart_pred = [0]
                distance, path = fastdtw(burststart_true, burststart_pred)
                burststart.append(distance)

#                 print(burststart_true)
                interarrival_true = np.diff(np.array(burststart_true))
                interarrival_pred = np.diff(np.array(burststart_pred))
#                 print(interarrival_true)
                interarrival_true = sum(interarrival_true)/len(interarrival_true) if len(interarrival_true)!=0 else 9900
                interarrival_pred = sum(interarrival_pred)/len(interarrival_pred) if len(interarrival_pred)!=0 else 9900
                burst_interarrival.append(abs(interarrival_true - interarrival_pred) / interarrival_true)

                burstduration_true = []
                burstduration_pred = []
                for i in bursts_index_true:
                    burstduration_true.append(len(i))
                for i in bursts_index_pred:
                    burstduration_pred.append(len(i))
                burstduration_true = sum(burstduration_true) / len(burstduration_true) \
                                                    if len(burstduration_true) != 0 else 0
                burstduration_pred = sum(burstduration_pred) / len(burstduration_pred) \
                                                    if len(burstduration_pred) != 0 else 0
                if burstduration_true == 0:
                    burstduration.append(abs(burstduration_true - burstduration_pred) / 1)
                else:
                    burstduration.append(abs(burstduration_true - burstduration_pred) / burstduration_true)

                burstvol_true = []
                burstvol_pred = []
                for i in bursts_val_true:
                    burstvol_true.append(sum(i))
                for i in bursts_val_pred:
                    burstvol_pred.append(sum(i))
                burstvol_true = sum(burstvol_true)/len(burstvol_true) if len(burstvol_true) != 0 else 0
                burstvol_pred = sum(burstvol_pred)/len(burstvol_pred) if len(burstvol_pred) != 0 else 0
                if burstvol_true == 0:
                    burstvol.append(abs(burstvol_true - burstvol_pred) / 1)
                else:    
                    burstvol.append(abs(burstvol_true - burstvol_pred) / burstvol_true)

            mse_perqueue += sum(mse) / len(mse)
            acorr_perqueue += sum(acorr) / len(acorr)
            zero_perqueue += sum(zero) / len(zero)
            p90_perqueue += sum(p90) / len(p90)
            p99_perqueue += sum(p99) / len(p99)
            emd_perqueue += sum(emd) / len(emd)
            num_bursts_perqueue += sum(num_bursts) / len(num_bursts)
            burstmax_perqueue += sum(burstmax) / len(burstmax)
            burststart_perqueue += sum(burststart) / len(burststart)
            burstvol_perqueue += sum(burstvol) / len(burstvol)
            burst_interarrival_perqueue += sum(burst_interarrival) / len(burst_interarrival)
            burstduration_perqueue += (sum(burstduration) / len(burstduration))
            # add += 1

    result['MSE'].append(round(mse_perqueue/num_queue,3))
    result['Burst_start_pos'].append(round(burststart_perqueue/num_queue,3))
    result['Burst_height'].append(round(burstmax_perqueue/num_queue,3))
    result['Burst_freq'].append(round(num_bursts_perqueue/num_queue,3))
    result['Burst_interarrival'].append(round(burst_interarrival_perqueue/num_queue,3))
    result['Burst_duration'].append(round(burstduration_perqueue/num_queue,3))
    result['Burst_volume'].append(round(burstvol_perqueue/num_queue,3))
    result['Empty_queues'].append(round(zero_perqueue/num_queue,3))
    result['90_percentile'].append(round(p90_perqueue/num_queue,3))
    result['99_percentile'].append(round(p99_perqueue/num_queue,3))
    result['emd'].append(round(emd_perqueue/num_queue,3))
#     res['KS_test'].append(round(eight/num_queue,5))
    result['Autocorrelation'].append(round(acorr_perqueue/num_queue,3))
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

def burst_detection(data, th, low_threshold, high_threshold):
    res = []
    bursts_val = []
    bursts_index = []
    height = []
    inc_rate = []
    dec_rate = []
    peak_pos = []
    burst = False
    for i in range(len(data)):
        temp = []
        temp_ind = []
        if burst == False and data[i] > low_threshold:
            temp.append(data[i])
            temp_ind.append(i)
            j = i 
            while j + 1 < len(data) -1:
                j = j + 1
                if data[j] > low_threshold:
                    temp.append(data[j])
                    temp_ind.append(j)
                else:
                    dec = j
                    dec_v = data[j]
                    break
            if len(temp) > 1:
                ind = temp.index(max(temp))
                if (ind > 0 and (temp[ind] - temp[0])/ind > th and temp[ind] > high_threshold) \
                    or (ind == 0 and temp[ind] > high_threshold):
                    burst = True
                    bursts_val.append(temp)
                    bursts_index.append(temp_ind)
            elif len(temp) == 1 and temp[0] > high_threshold:
                bursts_val.append(temp)
                bursts_index.append(temp_ind)
        elif burst == True and data[i] < low_threshold:
            burst = False
    return bursts_val, bursts_index