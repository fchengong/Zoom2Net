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
        
        for features, labels in iter(dataset):
            features = features.float().cuda()
            labels = labels.float().cuda()
            result = model(features)
            loss = criterion(result, labels)\
                +config.emd_weight*ot.wasserstein_1d(result[0], labels[0])

            for i in range(num_window):
                l1 = (labels[:,coarse*i:coarse*(i+1)].max(-1)[0] - \
                    result[:,coarse*i:coarse*(i+1)].max(-1)[0])
                loss += mu * torch.sum(l1**2)

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
    for features, labels, set_fine_grain, l, l_batch_max, l_batch_period, l_batch_inequal in train_iter:
        print(f"- Batch set #{train_count}.", end='')
        features = features.float().cuda()
        labels = labels.float().cuda()
        l_batch_max = l_batch_max.float().cuda()
        l_batch_period = l_batch_period.float().cuda()
        l_batch_inequal = l_batch_inequal.float().cuda()
        set_fine_grain = set_fine_grain.float().cuda()
        model.zero_grad()

        imputed_time_series = model(features)
        loss = 0

        for i in range(features.shape[0]):
            index = l[i]
            a = set_fine_grain[i,:index] - imputed_time_series[i]
            mse, ind = torch.min(torch.mean(a**2,dim=-1),dim=0, keepdim=False)
            loss += mse + config.emd_weight*ot.wasserstein_1d(imputed_time_series[i], set_fine_grain[i,ind])

        constr_square = 0
        constr_l1 = 0
        ineq_l1 = 0
        ineq_square = 0
        l_period_square = 0
        period_l1 = 0
        # Incorporate constriant violations
        for i in range(num_window):
            l_max = labels[:,COARSE*i:COARSE*(i+1)].max(-1)[0] - \
                    imputed_time_series[:,COARSE*i:COARSE*(i+1)].max(-1)[0]
            constr_square += mu * torch.sum(l_max**2)
            constr_l1 += torch.sum(l_batch_max[:,i,:] * l_max)

        loss += constr_square + constr_l1  + ineq_l1 + ineq_square + l_period_square+ period_l1 
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
    num_window = int(WINDOW_SIZE / COARSE)   # 6
    periods = np.arange(0, WINDOW_SIZE, int(COARSE/2))
    check_constr_l1 = 0
    max_constr = 0
    check_count = 0
    update_train_dataset = []
    train_iter = iter(train_loader) 
    burst_threshold = (torch.ones(1)*0.5).cuda()
    model.eval()
    with torch.no_grad():
        for features, labels, set_fine_grain, l, l_batch_max, l_batch_period, l_batch_inequal in train_iter:
            features = features.float().cuda()
            labels = labels.float().cuda()
            l_batch_max = l_batch_max.float().cuda()
            l_batch_period = l_batch_period.float().cuda()
            l_batch_inequal = l_batch_inequal.float().cuda()
            result = model(features)
            
            for i in range(num_window):                
                l1 = (labels[:,COARSE*i:COARSE*(i+1)].max(-1)[0] - \
                    result[:,COARSE*i:COARSE*(i+1)].max(-1)[0])
                max_constr += torch.sum(torch.abs(l1))
                l_batch_max[:,i,0] += 2 * mu * l1
            
            check_count += 1
            for i in range(features.shape[0]):
                update_train_dataset.append((features[i].detach().cpu().numpy(), labels[i].detach().cpu().numpy(), \
                                         set_fine_grain[i].detach().cpu().numpy(), l[i].detach().cpu().numpy(), \
                                         l_batch_max[i].detach().cpu().numpy(),\
                                         l_batch_period[i].detach().cpu().numpy(), l_batch_inequal[i].detach().cpu().numpy()))
        check_constr_l1 = max_constr
    return check_constr_l1, max_constr, update_train_dataset

# Get constraint violations for testing data
def test_constraint(model, test_dataset, tanh_scaling = 500, WINDOW_SIZE = 1000, COARSE = 10, device=torch.device("cpu")):
    max_constraint_violation = 0

    with torch.no_grad():
        for i in range(len(test_dataset)):
            x = inference(model, test_dataset[i][0], WINDOW_SIZE, COARSE, device)[0].cpu().numpy()
            for j in range(int(WINDOW_SIZE / COARSE)):
                max_constraint_violation += abs(test_dataset[i][1][COARSE*j:COARSE*(j+1)].max() - \
                    x[COARSE*j:COARSE*(j+1)].max(-1))

    return max_constraint_violation

def inference_withoutCCA(
    model: TSTransformerEncoder,
    datapoint: np.ndarray,
    WINDOW_SIZE = 300, COARSE = 50, device=torch.device("cpu")
):
    with torch.no_grad():
        features = torch.empty((1, 2, 6, int(WINDOW_SIZE / COARSE)), device=device)
        features[0, :, :,:] = torch.from_numpy(datapoint)
        imputed_time_series = model(features[:,:,:])
        return imputed_time_series

def inference(
    model: TSTransformerEncoder,
    datapoint: np.ndarray,
    WINDOW_SIZE = 300, COARSE = 50, device=torch.device("cpu")
):
    with torch.no_grad():
        features = torch.empty((1, 2, 7, int(WINDOW_SIZE / COARSE)), device=device)
        features[0, :, :,:] = torch.from_numpy(datapoint)
        imputed_time_series = model(features[:,:,:])
        return imputed_time_series

def convert_even_plain(data):
    a = np.zeros((8,6,6))
    for i in range(8):
        for j in range(6):
            a[i,0,j] = data[i*2][0][j*50]
            a[i,1,j] = data[i*2][3][j*50]
            a[i,2,j] = data[i*2+1][0][j*50]
            a[i,3,j] = data[i*2+1][3][j*50]
            a[i,4,j] = data[i*2][1][j*50]
            a[i,5,j] = data[i*2][2][j*50]
    return a

def convert_odd_plain(data):
    a = np.zeros((8,6,6))
    for i in range(8):
        for j in range(6):
            a[i,0,j] = data[i*2][0][j*50] 
            a[i,1,j] = data[i*2][3][j*50]
            a[i,2,j] = data[i*2+1][0][j*50]
            a[i,3,j] = data[i*2+1][3][j*50]
            a[i,4,j] = data[i*2][1][j*50]
            a[i,5,j] = data[i*2][2][j*50]
    return a