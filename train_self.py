
# =============================================================================
# Packages and settings
# =============================================================================

# load packages
import sys
import numpy as np
import torch
import torch.nn as nn
import pickle as pkl
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
import logging
import os.path as osp
import os
from datetime import datetime
import json
# some settings
np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# Fourier Neural Operator (Reference: https://arxiv.org/pdf/2010.08895.pdf)
# =============================================================================

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        2D Fourier layer: FFT -> Linear Transform -> Inverse FFT  
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    def forward(self, x):
        # Perform Fourier transform
        batchsize = x.shape[0]
        x_ft = torch.rfft(x, 2, normalized=True, onesided=True)

        # Multiply top Fourier modes with Fourier weights
        out_ft = torch.zeros(batchsize, self.in_channels,  x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Perform Inverse Fourier transform
        x = torch.irfft(out_ft, 2, normalized=True, onesided=True, signal_sizes=( x.size(-2), x.size(-1)))
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(SimpleBlock2d, self).__init__()
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        # Projection P
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # FNO Layer 1
        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)
        # FNO Layer 2
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)
        # FNO Layer 3
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)
        # FNO Layer 4
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)
        # Projection Q
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()
        """
        A wrapper function
        """
        self.conv1 = SimpleBlock2d(modes1, modes2,  width)

    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c

# =============================================================================
# Some useful functions
# =============================================================================

# Relative l-2 norm loss function
class DataLoss(object):
    def __init__(self, p=2):
        super(DataLoss, self).__init__()
        self.p = p

    def data_loss(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        return torch.mean(diff_norms/y_norms)

    def __call__(self, x, y):
        return self.data_loss(x, y)


# Complex multiplication
def compl_mul2d(a, b):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    op = partial(torch.einsum, "bixy,ioxy->boxy")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

# initiale weights
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if type(m) == torch.nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# random masking of boundary values
def rand_mask(x_test, mp=0):
    if mp==0:
        return x_test
    else:
        mk=int(mp*x_test.shape[1])
        mind=np.random.randint(0,x_test.shape[1],size=(x_test.shape[0],mk))
        for i in range(mind.shape[0]):
            x_test[...,0][i,mind[i,:]]=-1
            x_test[...,-1][i,mind[i,:]]=-1
        return x_test

def load_data(f_names_train, f_names_test, data_fold_train, data_fold_test, b_size, ntest_sc, ntest_max, start_t, end_t, K_pred, K_pred_test, train=True, val=True,
              normalize=True, **kwargs):

    # load test data
    x_test = []; y_test = []
    if val == True:
        for f in f_names_train:
            with open(data_fold_train + f'train_{f}.pkl', 'rb') as f:
                data = pkl.load(f)
                x_test.append(data['X'][ntrain_sc:ntest_sc])
                y_test.append(data['Y'][ntrain_sc:ntest_sc])
    else:
        for f in f_names_test:
            with open(data_fold_test + f'test_{f}.pkl', 'rb') as f:
                data = pkl.load(f)
                x_test.append(data['X'][:])
                y_test.append(data['Y'][:])
    x_test = np.concatenate(x_test, axis=0).astype(np.float32)
    y_test = np.concatenate(y_test, axis=0).astype(np.float32)
    x_test = x_test[:ntest_max,start_t:end_t,:]
    y_test = y_test[:ntest_max,start_t:end_t,:]
    x_test[:,1:,1:-1] = -1
    if len(K_pred_test) > 0:
        x_test[:, 0:1, 1:-1] = K_pred_test[:, -1:, 1:-1]
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    # grid size params
    s1 =  x_test.shape[1]
    s2 = x_test.shape[2]
    ntest = x_test.shape[0]

    # load train data if required
    ntrain = 0
    # if train:
    x_train = []; y_train = []
    for f in f_names_train:
        with open(data_fold_train+f'train_{f}.pkl','rb') as f:
            data = pkl.load(f)
        x_train.append(data['X'][:ntrain_sc])
        y_train.append(data['Y'][:ntrain_sc])
    x_train = np.concatenate(x_train, axis=0).astype(np.float32)
    y_train = np.concatenate(y_train, axis=0).astype(np.float32)
    x_train = x_train[:ntrain_max,start_t:end_t,:]
    y_train = y_train[:ntrain_max,start_t:end_t,:]
    x_train[:,1:,1:-1] = -1
    if len(K_pred) > 0:
        x_train[:, 0:1, 1:-1] = K_pred[:, -1:, 1:-1]
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    ntrain = x_train.shape[0]

    # concat location coordinates
    grids = []

    grids.append(np.linspace(0, s1, s1))
    grids.append(np.linspace(0, s2,s2))

    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    # s1 = end_t - start_t
    grid = grid.reshape(1,s1,s2,2)
    grid = torch.tensor(grid, dtype=torch.float)
    x_test = torch.cat([x_test.reshape(ntest,s1,s2,1),
                        grid.repeat(ntest,1,1,1)], dim=3)

    # pytorch loader
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=b_size, shuffle=False)
    train_loader = None
    if train:
        x_train = torch.cat([x_train.reshape(ntrain,s1,s2,1),
                             grid.repeat(ntrain,1,1,1)], dim=3)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train),
            batch_size=b_size, shuffle=True)
    else:
        x_train = torch.cat([x_train.reshape(ntrain,s1,s2,1),
                             grid.repeat(ntrain,1,1,1)], dim=3)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train),
            batch_size=b_size, shuffle=False)

    return train_loader, test_loader, ntrain, ntest, s1, s2



def freeze_parameters_except_conv1d(model):
    for name, param in model.named_parameters():
        if 'w' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def get_optimizer(model, learning_rate):
    return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)

def get_logger(save_path):
    os.makedirs(save_path, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    hdlr = logging.FileHandler(osp.join(save_path, 'log.txt'))
    console = logging.StreamHandler()
    fmtr = logging.Formatter('%(message)s')
    hdlr.setFormatter(fmtr)
    console.setFormatter(fmtr)

    logger.addHandler(hdlr)
    logger.addHandler(console)

    return logger

step = 100
batch_size = 64
layers = 4
width = 64
patience = 100
modes1 = 24
modes2 = modes1

save_path = "../models/" + "self/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

logger = get_logger(save_path)
logger.info('Start time: {}'.format(datetime.now()))

logger.info('--------- Model Info ---------')
logger.info('Model name: ' + 'self_bvp')
# =============================================================================
# Parameters
# =============================================================================

learning_rate = 1e-3
epochs = 500
step_size = 100
gamma = 0.5
lam = 2.0

ntrain_sc = 1300
ntest_sc = 50 + ntrain_sc
ntrain_max = 10000
ntest_max = 500
eval_batch_size = 1
# =============================================================================
# Load data
# =============================================================================

print('\n ------ Loading dataset -------')

# data files
train_req = True

df_test = '../data/test/'
df_train = '../data/train/'
f_names_train = ['lwrg20x1gs-sic1','lwrg20x1gs-sic2','lwrg20x1gs-sic3']
f_names_test =  ['lwrg20x1gs-random','lwrg20x1gs-random_w1signal','lwrg20x1gs-random_w2signal','lwrg20x1gs-random_w3signal']

# data params

# =============================================================================
# Training FNO-2D model
# =============================================================================
logger.info('---------- Training ----------')
start = 0

stop = 600
steps_lb = np.arange(0, stop + step, step)
steps_ub = steps_lb+1

s1, s2 = step, 50

K_train_arr = np.zeros((2, 3900, stop, s2))
K_test_arr = np.zeros((2, 200, stop, s2))

counter = 0
train_time1 = default_timer()
train_dataarr=[]; train_physarr=[]
test_dataarr=[]; test_physarr=[]
for i in range(0, steps_lb.size - 1):
    start_t = steps_lb[i]
    end_t = steps_ub[i + 1]
    print(start_t, end_t)

    if counter == 0:
        print('Training the FNO model...\n')

        K_pred = []
        K_pred_test = []
        # train batch loader
        train_loader, test_loader, ntrain, ntest, s1, s2 = load_data(
            f_names_train, f_names_test, df_train, df_test, batch_size, ntest_sc, ntest_max, start_t=start_t, end_t=end_t, K_pred= K_pred, K_pred_test= K_pred_test,
            train=True, ntrain_sc=ntrain_sc, ntrian_max=ntrain_max)
        logger.info('number of train samples: {}'.format(ntrain))
        logger.info('number of ntest samples: {}'.format(ntest))
        logger.info('grid resolution of train: ' + str(s1) + ' x ' + str(s2))
        # 初始化模型
        model = FNO2d(modes1, modes2, width)
        model.apply(init_weights)
        model.to(device)


        best_val_loss = float('inf')
        patience_counter = 0


        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        # initialize
        data_loss = DataLoss()

        t1_train = default_timer()
        # training loop
        for ep in range(1,epochs+1):
            # training and validation

            model.train()
            train_dataloss = 0

            for x, y in train_loader:
                # initialize
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                # forward pass
                out = model(x)
                # backward pass
                d_loss = data_loss(y, out)
                loss = d_loss
                loss.backward()
                # update
                optimizer.step()
                train_dataloss += d_loss.item()
            train_dataloss /= len(train_loader)

            # update learning rate
            scheduler.step()
            train_dloss = train_datalos

            model.eval()
            test_dataloss = 0.0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    test_dataloss += data_loss(out, y).item()

            test_dataloss /= len(test_loader)

            test_dloss = test_dataloss

            train_dataarr.append(train_dloss)
            test_dataarr.append(test_dloss)


            if test_dloss < best_val_loss:
                best_val_loss = test_dloss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

            t2_train = default_timer()
            logger.info(f'i: {i}, ep: {ep}, {t2_train - t1_train:.03f}, train_dloss: {train_dloss:.06f}, '
                        f'test_dloss: {test_dloss:.06f}, best_val_loss: {best_val_loss:.06f}, patience_counter: {patience_counter:.00f}')


        logger.info('---------- transfer training data  ----------')
        ev_train_loader, test_loader, ntrain, ntest, s1, s2 = load_data(
            f_names_train, f_names_test, df_train, df_test, eval_batch_size, ntest_sc, ntest_max, start_t=start_t,
            end_t=end_t, K_pred=K_pred, K_pred_test= K_pred_test, train=False,
            ntrain_sc=ntrain_sc, ntrian_max=ntrain_max)

        model.eval()
        index = 0
        pred = torch.zeros((ntrain, s1, s2))
        act = torch.zeros((ntrain, s1, s2))
        with torch.no_grad():
            for x, y in ev_train_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred[index] = out
                act[index] = y.squeeze(0)
                index = index + 1

        K_pred = pred.cpu().numpy()
        K_act = act.cpu().numpy()

        K_train_arr[0, :, start_t:end_t, :] = K_pred
        K_train_arr[1, :, start_t:end_t, :] = K_act

        # save offline
        print('\n ------ Saving model offline -------')
        torch.save(model.state_dict(), save_path+str(counter)+'_best_model.pt')

    else:
        logger.info('Training the transfer FNO model...\n')

        # train batch loader
        train_loader, test_loader, ntrain, ntest, s1, s2 = load_data(
            f_names_train, f_names_test, df_train, df_test, batch_size, ntest_sc, ntest_max, start_t=start_t, end_t=end_t, K_pred=K_pred, K_pred_test= K_pred_test,
            train=True, ntrain_sc=ntrain_sc, ntrian_max=ntrain_max)


        model_transfer = FNO2d(modes1, modes2, width)
        model_transfer.load_state_dict(torch.load(save_path+str(counter-1)+'_best_model.pt'))
        model_transfer.to(device)


        best_val_loss = float('inf')
        patience_counter = 0


        freeze_parameters_except_conv1d(model_transfer)


        optimizer_transfer = get_optimizer(model_transfer, learning_rate)
        scheduler_transfer = torch.optim.lr_scheduler.StepLR(optimizer_transfer, step_size=step_size, gamma=gamma)

        # initialize
        data_loss = DataLoss()
        t1_train = default_timer()

        # training loop
        for ep in range(1, epochs + 1):
            # train_dloss, train_ploss = train(train_loader)

            model_transfer.train()
            train_dataloss = 0

            for x, y in train_loader:
                # initialize
                x, y = x.to(device), y.to(device)
                optimizer_transfer.zero_grad()
                # forward pass
                out = model_transfer(x)
                # backward pass
                d_loss = data_loss(y, out)

                loss = d_loss
                loss.backward()
                # update
                optimizer_transfer.step()
                train_dataloss += d_loss.item()
            train_dataloss /= len(train_loader)

            # update learning rate
            scheduler_transfer.step()
            train_dloss = train_dataloss
            train_dataarr.append(train_dloss)


            model_transfer.eval()
            test_dataloss = 0.0

            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    out = model_transfer(x)
                    test_dataloss += data_loss(out, y).item()

            test_dataloss /= len(test_loader)

            test_dloss = test_dataloss

            train_dataarr.append(train_dloss)
            test_dataarr.append(test_dloss)


            if test_dloss < best_val_loss:
                best_val_loss = test_dloss
                patience_counter = 0
                torch.save(model_transfer.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

            t2_train = default_timer()
            logger.info(f'i: {i}, ep: {ep}, {t2_train - t1_train:.03f}, train_dloss: {train_dloss:.06f}, '
                        f'test_dloss: {test_dloss:.06f}, best_val_loss: {best_val_loss:.06f}, patience_counter: {patience_counter:.00f}')

        logger.info('---------- transfer training data ----------')

        ev_train_loader, test_loader, ntrain, ntest, s1, s2 = load_data(
            f_names_train, f_names_test, df_train, df_test, eval_batch_size, ntest_sc, ntest_max, start_t=start_t,
            end_t=end_t, K_pred=K_pred, K_pred_test= K_pred_test, train=False,
            ntrain_sc=ntrain_sc, ntrian_max=ntrain_max)

        model_transfer.eval()
        index = 0
        pred = torch.zeros((ntrain, s1, s2))
        act = torch.zeros((ntrain, s1, s2))
        with torch.no_grad():
            for x, y in ev_train_loader:
                x, y = x.to(device), y.to(device)
                out = model_transfer(x)
                pred[index] = out
                act[index] = y.squeeze(0)
                index = index + 1

        K_pred = pred.cpu().numpy()
        K_act = act.cpu().numpy()

        K_train_arr[0, :, start_t:end_t, :] = K_pred
        K_train_arr[1, :, start_t:end_t, :] = K_act

        # save offline
        print('\n ------ Saving model offline -------')
        torch.save(model_transfer.state_dict(), save_path+str(counter)+'_best_model.pt')

    counter += 1

train_time2 = default_timer()
logger.info(f'total time: {train_time2-train_time1:.06f}')
# Testing results
# =============================================================================
logger.info('---------- final test ----------')

counter = 0
test_time1 = default_timer()

K_pred = []
K_pred_test = []

for i in range(0, steps_lb.size - 1):
    start_t = steps_lb[i]
    end_t = steps_ub[i + 1]
    print(start_t, end_t)

    # train batch loader
    train_loader, test_loader, ntrain, ntest, s1, s2 = load_data(
        f_names_train, f_names_test, df_train, df_test, eval_batch_size, ntest_sc, ntest_max, start_t=start_t, end_t=end_t,
        K_pred=K_pred, K_pred_test=K_pred_test,
        train=False, val=False, ntrain_sc=ntrain_sc, ntrian_max=ntrain_max)

    # 加载模型
    model = FNO2d(modes1, modes2, width)
    model.load_state_dict(torch.load(save_path + str(counter) + '_best_model.pt'))
    model.to(device)

    index = 0
    pred_test = torch.zeros((ntest, s1, s2))
    act_test = torch.zeros((ntest, s1, s2))
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred_test[index] = out
            act_test[index] = y.squeeze(0)
            index = index + 1

    K_pred_test = pred_test.cpu().numpy()
    K_act_test = act_test.cpu().numpy()

    K_test_arr[0, :, start_t:end_t, :] = K_act_test
    K_test_arr[1, :, start_t:end_t, :] = K_pred_test

counter += 1

test_time2 = default_timer()
logger.info(f'test time: {test_time2-test_time1:.06f}')
K_test_arr_act = K_test_arr[0]
K_test_arr_pred = K_test_arr[1]

K_rmse = np.sqrt(np.mean(np.power(K_test_arr_act-K_test_arr_pred, 2), axis=(1,2)))
K_mae = np.mean(np.abs(K_test_arr_act-K_test_arr_pred), axis=(1,2))

# print results

logger.info("###############     total_result:        ")
total_rmse=np.sqrt(np.mean(np.power(K_test_arr[0]-K_test_arr[1], 2)))
total_mae=np.mean(np.abs(K_test_arr[0]-K_test_arr[1]))
relative_error=np.linalg.norm(K_test_arr[0].flatten() - K_test_arr[1].flatten(), 2) / np.linalg.norm(K_test_arr[0].flatten(), 2)
logger.info(f'K_rmse: {total_rmse:0.06f} , K_mae: {total_mae:0.06f} , relative_error: {relative_error:0.06f} ')

np.save(save_path+'train_dataarr.npy', train_dataarr)
np.save(save_path+'test_dataarr.npy', test_dataarr)
np.save(save_path+'test_results.npy', K_test_arr)


logger.info('\n ------ Training and Evalution done -------')