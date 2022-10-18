import torch
from lib.dataset import TrafficDataset
from predictor import Predictor
from config import chicago_config, nyc_config, seq_len, pred_len
from torch.utils.data import DataLoader

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


cities = ['nyc', 'chicago']
configs = {'nyc': nyc_config, 'chicago': chicago_config}

# basic setting
city = cities[0]  # 0 for nyc, 1 for chicago
config = configs[city]
interval = 30

batch_size = 64
num_epochs = 500
lr = 0.001
weight_delay = 0.0001
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')


# data load
train_set = TrafficDataset(city, interval, seq_step=seq_len, pred_step=pred_len, mode='train')
train_loader = DataLoader(train_set, batch_size, shuffle=True)
val_set = TrafficDataset(city, interval, seq_step=seq_len, pred_step=pred_len, mode='val')
val_loader = DataLoader(val_set, batch_size, shuffle=False)

# model configuration
model = 'tsin'
model_conf = config[model]
predictor = Predictor(predefined_adj=None, model=model, model_args=model_conf,
                      uncertainty_weighting=True).to(device)

nParams = sum([p.nelement() for p in predictor.network.parameters()])
print('Number of model parameters is', nParams)

# model training
predictor.train_exec(num_epochs, lr, weight_delay, train_loader, val_loader, device)