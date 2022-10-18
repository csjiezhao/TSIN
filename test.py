import torch
from predictor import Predictor
from config import chicago_config, nyc_config, seq_len, pred_len
from lib.dataset import TrafficDataset
from torch.utils.data import DataLoader
from lib.metric_utils import metrics


cities = ['nyc', 'chicago']
configs = {'nyc': nyc_config, 'chicago': chicago_config}

# basic setting
city = cities[0]  # 0 for nyc, 1 for chicago
config = configs[city]
interval = 30
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

# data load
test_set = TrafficDataset(city, interval, seq_step=seq_len, pred_step=pred_len, mode='test')
test_loader = DataLoader(test_set, len(test_set), shuffle=False)

# model configuration & load
model = 'tsin'
model_conf = config[model]

predictor = Predictor(predefined_adj=None, model=model, model_args=model_conf, uncertainty_weighting=False).to(device)
predictor.load_model('', device=device)

# model test
taxi_outputs, taxi_targets, ride_outputs, ride_targets = predictor.test_exec(test_loader, device)
print('taxi:', metrics(taxi_outputs, taxi_targets))
print('ride:', metrics(ride_outputs, ride_targets))

