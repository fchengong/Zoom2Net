import ot
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from model_training.transformer import TSTransformerEncoder
import numpy as np

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def test_dataset_loss(model, criterion, dataset: DataLoader, config, mu = 0.1, tanh_scaling = 500):
    with torch.no_grad():
        total_loss = 0
        total_count = 0
        WINDOW_SIZE = config.window_size
        coarse = config.zoom_in_factor
        num_window = int(WINDOW_SIZE / coarse)   # 6
        periods = np.arange(0, WINDOW_SIZE, int(coarse/2))
        burst_threshold = (torch.ones(1)*0.5).cuda()
        
        for features, labels, time in iter(dataset):
            features = features.float().cuda()
            labels = labels.float().cuda()
            time = time.float().cuda()
            result = model(features)
            loss = criterion(result, labels)\
                +config.emd_weight*ot.wasserstein_1d(result[0], labels[0])

            for i in range(labels.shape[0]):
                start = 0
                for j in (time[i]):
                    if j == 0:
                        break
                    # print(type(i), type(j), type(start))
                    j = j.int()
                    l_max = labels[i, start:j].max(-1)[0] - \
                        result[i, start:j].max(-1)[0]
                    loss += mu * (l_max**2)
                    l_sum = labels[i, start:j].sum() - \
                            result[i, start:j].sum()
                    loss += mu/100 * (l_sum**2)
                    start = j

            total_loss += loss
            total_count += 1

        return total_loss / total_count
    
def train_epoch(
    model: nn.Module,
    train_dataset: DataLoader,
    val_dataset: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau, config, \
    batch_size = 16, mu = 0.1, tanh_scaling  = 500
):
    criterion = nn.MSELoss()
    WINDOW_SIZE = config.window_size
    COARSE = config.zoom_in_factor
    num_window = int(WINDOW_SIZE / COARSE)   # 6
    
    model.train()
    train_iter = iter(train_dataset)
    train_loss = 0
    train_count = 0
    processed_data = 0
    periods = np.arange(0, WINDOW_SIZE, int(COARSE))
    mse = 0
    for features, labels, set_fine_grain, l, l_batch_max, l_batch_sum, time in train_iter:
        print(f"- Batch set #{train_count}.", end='')
        features = features.float().cuda()
        labels = labels.float().cuda()
        time = time.int().cuda()
        l_batch_max = l_batch_max.float().cuda()
        l_batch_sum = l_batch_sum.float().cuda()
        set_fine_grain = set_fine_grain.float().cuda()
        model.zero_grad()

        imputed_time_series = model(features)
        loss = 0
        for i in range(features.shape[0]):
            index = l[i]
            a = set_fine_grain[i,:index] - imputed_time_series[i]
            mse, ind = torch.min(torch.mean(a**2,dim=-1),dim=0, keepdim=False)
            loss += mse + config.emd_weight*ot.wasserstein_1d(imputed_time_series[i], set_fine_grain[i,ind])

        sum_square = 0
        sum_l1 = 0
        max_square = 0
        max_l1 = 0
        # Incorporate constriant violations
        for i in range(labels.shape[0]):
            start = 0
            cnt = 0
            for j in (time[i]):
                if j == 0:
                    break
                l_max = labels[i, start:j].max(-1)[0] - \
                        imputed_time_series[i, start:j].max(-1)[0]
                max_square += mu * (l_max**2)
                max_l1 += (l_batch_max[i,cnt] * l_max)
                l_sum = labels[i, start:j].sum() - \
                        imputed_time_series[i, start:j].sum()
                sum_square += mu/100 * (l_sum**2)
                sum_l1 += (l_batch_sum[i,cnt]/15 * l_sum)
                start = j
                cnt += 1

        loss += max_square + max_l1 + sum_square + sum_l1
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.detach().cpu()
        train_count += 1
        processed_data += imputed_time_series.shape[0]
        print(f"Training Loss {loss}")
    print()

    # Validation
    validation_loss = test_dataset_loss(model, criterion, val_dataset, config, mu=mu)

    # Write sum
    train_loss = train_loss / train_count
    print(f"Training Loss {train_loss} Validation Loss {validation_loss} ")
    scheduler.step(train_loss)

    return scheduler, train_loss, validation_loss

# Get constraint violations for training dataset and do lagrangian updates
def check_constraints(model, train_loader, mu=0.01, batch_size=16, tanh_scaling = 500, WINDOW_SIZE = 1000, COARSE = 50):
    check_constr_l1 = 0
    sum_constr = 0
    max_constr = 0
    check_count = 0
    update_train_dataset = []
    train_iter = iter(train_loader) 
    model.eval()
    with torch.no_grad():
        for features, labels, set_fine_grain, l, l_batch_max, l_batch_sum, time in train_iter:
            features = features.float().cuda()
            labels = labels.float().cuda()
            time = time.int().cuda()
            l_batch_max = l_batch_max.float().cuda()
            l_batch_sum = l_batch_sum.float().cuda()
            result = model(features)
            
            # print(f"labels.shape[0]: {labels.shape[0]}")
            for i in range(labels.shape[0]):
                # print(f"time[i]: {time[i]}")
                start = 0
                cnt = 0
                for j in (time[i]):
                    if j == 0:
                        break
                    l_max = labels[i, start:j].max(-1)[0] - \
                        result[i, start:j].max(-1)[0]
                    max_constr += (torch.abs(l_max))
                    l_batch_max[i,cnt] += 2 * mu * l_max
                    l_sum = labels[i, start:j].sum() - \
                            result[i, start:j].sum()
                    sum_constr += (torch.abs(l_sum))
                    l_batch_sum[i,cnt] += 2 * mu * l_sum
                    start = j
                    cnt += 1
            
            check_count += 1
            for i in range(features.shape[0]):
                update_train_dataset.append((features[i].detach().cpu().numpy(), labels[i].detach().cpu().numpy(), \
                                         set_fine_grain[i].detach().cpu().numpy(), l[i].detach().cpu().numpy(), \
                                         l_batch_max[0].detach().cpu().numpy(), l_batch_sum[0].detach().cpu().numpy(), \
                                         time[0].detach().cpu().numpy()))
        check_constr_l1 = max_constr + sum_constr
    return check_constr_l1, update_train_dataset

# Get constraint violations for testing data
def test_constraint(model, test_dataset, tanh_scaling = 500, WINDOW_SIZE = 300, COARSE = 10, device=torch.device("cpu")):
    model.eval()
    constr_vio_sum = 0
    constr_vio_max = 0

    with torch.no_grad():
        for i in range(len(test_dataset)):
            x = inference(model, (test_dataset[i][0]), WINDOW_SIZE, device)[0].cpu().numpy()
            start = 0
            cnt = 0
            for j in (test_dataset[i][-1].astype(int)):
                if j == 0:
                    break
                constr_vio_max += (np.abs(np.max(test_dataset[i][1][start:j]) - np.max(x[start:j])))
                constr_vio_sum += (np.abs(np.sum(test_dataset[i][1][start:j]) - np.sum(x[start:j])))
                start = j
                cnt += 1

    return constr_vio_max + constr_vio_sum

def inference(
    model: TSTransformerEncoder,
    datapoint: np.ndarray,
    WINDOW_SIZE = 300, device=torch.device("cpu")
):
    with torch.no_grad():
        features = torch.empty((1, 7, WINDOW_SIZE), device=device)
        features[0, :, :] = torch.from_numpy(datapoint)
        imputed_time_series = model(features[:,:,:])
        return imputed_time_series
