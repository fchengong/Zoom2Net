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
        COARSE = config.zoom_in_factor
        num_window = int(WINDOW_SIZE / COARSE)   # 6
        periods = np.arange(0, WINDOW_SIZE, int(COARSE/2))
        burst_threshold = (torch.ones(1)*0.5).cuda()
        
        for features, labels in iter(dataset):
            features = features.float().cuda()
            labels = labels.float().cuda()
            result = model(features)[:,0,:]
            loss = criterion(result, labels)\
                +config.emd_weight*ot.wasserstein_1d(result[0], labels[0])
            for i in range(num_window):
                l_sum = torch.sum(labels[:,COARSE*i:COARSE*(i+1)], dim=-1) - \
                        torch.sum(result[:,COARSE*i:COARSE*(i+1)], dim=-1)
                loss += mu/100 * torch.sum(l_sum**2)

                retrans = features[:,0,2,i]
                convert = torch.tanh(retrans*tanh_scaling)
                l_retrans = (convert * ((result[:,i*COARSE:(i+1)*COARSE]).max(-1)[0] - burst_threshold))
                loss += torch.sum(abs(l_retrans))

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
    for features, labels, set_fine_grain, l, l_batch_sum, l_batch_retrans in train_iter:
        print(f"- Batch set #{train_count}.", end='')
        features = features.float().cuda()
        labels = labels.float().cuda()
        l_batch_sum = l_batch_sum.float().cuda()
        l_batch_retrans = l_batch_retrans.float().cuda()
        set_fine_grain = set_fine_grain.float().cuda()
        burst_threshold = (torch.ones(1)*0.5).cuda()
        model.zero_grad()

        imputed_time_series = model(features)[:,0,:]
        loss = 0

        loss = criterion(imputed_time_series, labels)
        for i in range(features.shape[0]):
            index = l[i]
            a = set_fine_grain[i,:index] - imputed_time_series[i]
            mse, ind = torch.min(torch.mean(a**2,dim=-1), dim=0, keepdim=False)
            loss += mse\
                + config.emd_weight*ot.wasserstein_1d(imputed_time_series[i], set_fine_grain[i,ind])

        sum_square = 0
        sum_l1 = 0
        retrans_square = 0
        retrans_l1 = 0
        # Incorporate constriant violations
        for i in range(num_window):
            l_sum = torch.sum(labels[:,COARSE*i:COARSE*(i+1)], dim=-1) - \
                    torch.sum(imputed_time_series[:,COARSE*i:COARSE*(i+1)], dim=-1)
            sum_square += mu/100 * torch.sum(l_sum**2)
            sum_l1 += torch.sum(l_batch_sum[:,i]/15 * l_sum)

            retrans = features[:,0,2,i] 
            convert = torch.tanh(retrans*tanh_scaling)
            l_retrans = (convert * ((imputed_time_series[:,i*COARSE:(i+1)*COARSE]).max(-1)[0] - burst_threshold))
            retrans_square += mu * torch.sum(l_retrans**2)
            retrans_l1 += torch.sum(l_batch_retrans[:,i] * l_retrans)

        loss += sum_square + sum_l1 + retrans_square + retrans_l1 
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
def check_constraints(model, train_loader, mu=0.01, batch_size=16, tanh_scaling = 500, WINDOW_SIZE = 1000, COARSE = 10):
    num_window = int(WINDOW_SIZE / COARSE)   # 6
    periods = np.arange(0, WINDOW_SIZE, int(COARSE/2))
    check_constr_l1 = 0
    sum_constr = 0
    retrans_constr = 0
    check_count = 0
    update_train_dataset = []
    train_iter = iter(train_loader) 
    burst_threshold = (torch.ones(1)*0.5).cuda()
    model.eval()
    with torch.no_grad():
        for features, labels, set_fine_grain, l, l_batch_sum, l_batch_retrans in train_iter:
            features = features.float().cuda()
            labels = labels.float().cuda()
            l_batch_sum = l_batch_sum.float().cuda()
            l_batch_retrans = l_batch_retrans.float().cuda()
            result = model(features)[:,0,:]
            
            for i in range(num_window):                
                l_sum = torch.sum(labels[:,COARSE*i:COARSE*(i+1)], dim=-1) - \
                    torch.sum(result[:,COARSE*i:COARSE*(i+1)], dim=-1)
                sum_constr += torch.sum(torch.abs(l_sum))
                l_batch_sum[:,i] += 2 * mu * l_sum

                retrans = features[:,0,2,i] 
                convert = torch.tanh(retrans*tanh_scaling)
                l_retrans = (convert * ((result[:,i*COARSE:(i+1)*COARSE]).max(-1)[0] - burst_threshold))
                retrans_constr += torch.sum(torch.abs(l_retrans))
                l_batch_retrans[:,i] += 2 * mu * l_retrans
            
            check_count += 1
            for i in range(features.shape[0]):
                update_train_dataset.append((features[i].detach().cpu().numpy(), labels[i].detach().cpu().numpy(), \
                                             set_fine_grain[i].detach().cpu().numpy(), l[i].detach().cpu().numpy(), \
                                             l_batch_sum[i].detach().cpu().numpy(), l_batch_retrans[i].detach().cpu().numpy()))
        check_constr_l1 = sum_constr + retrans_constr
        print(f"Sum constraint error: {sum_constr}")
    return check_constr_l1, sum_constr, retrans_constr, update_train_dataset

# Get constraint violations for testing data
def test_constraint(model, test_dataset, tanh_scaling = 500, WINDOW_SIZE = 1000, COARSE = 10, device=torch.device("cpu")):
    sum_constraint_violation = 0
    retrans_constraint_violation = 0

    with torch.no_grad():
        for i in range(len(test_dataset)):
            x = inference(model, test_dataset[i][0], WINDOW_SIZE, COARSE, device)[0][0].cpu().numpy()
            for j in range(int(WINDOW_SIZE / COARSE)):
                if np.sum(x[j*COARSE:(j+1)*COARSE]) < np.sum(test_dataset[i][1][j*COARSE:(j+1)*COARSE]):
                    sum_constraint_violation += np.sum(test_dataset[i][1][j*COARSE:(j+1)*COARSE]) - np.sum(x[j*COARSE:(j+1)*COARSE])
                retrans = test_dataset[i][0][0][2][j]
                convert = np.tanh(retrans*tanh_scaling)
                e = (convert * (np.max(x[j*COARSE:(j+1)*COARSE]) - 0.5))
                retrans_constraint_violation += np.abs(e)
    return sum_constraint_violation + retrans_constraint_violation

def inference(
    model: TSTransformerEncoder,
    datapoint: np.ndarray,
    WINDOW_SIZE = 1000, COARSE = 10, device=torch.device("cpu")
):
    with torch.no_grad():
        features = torch.empty((1, 2, 4, int(WINDOW_SIZE / COARSE)), device=device)
        features[0, :, :,:] = torch.from_numpy(datapoint)
        imputed_time_series = model(features[:,:,:])
        return imputed_time_series
