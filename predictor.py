import torch
import time
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from model.tsin import TSIN

MODEL_LIST = {'tsin': TSIN}


class Predictor(nn.Module):
    def __init__(self, predefined_adj, model, model_args, uncertainty_weighting):
        super(Predictor, self).__init__()
        self.network = MODEL_LIST[model](predefined_adj, model_args)
        self.network_init()
        self.model_name = str(type(self.network).__name__)
        self.loss_fun = nn.MSELoss()
        self.train_progress = []
        self.val_progress = []
        self.uncertainty_weighting = uncertainty_weighting

        self.sigma_1 = nn.Parameter(torch.ones(1))
        self.sigma_2 = nn.Parameter(torch.ones(1))

    def network_init(self):
        for param in self.network.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param)

    def get_optimizer(self, lr, wd):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    def forward(self, x):
        """
        :param x: (B, seq_len, N, 2 * num_feats)
        :return: taxi_pred, ride_pred
        """
        return self.network(x)

    def train_exec(self, num_epochs, lr, wd, train_loader, val_loader, device):
        print('Training on', device)
        best_epoch = 0
        best_val_loss = np.inf
        start_time = time.time()
        optimizer = self.get_optimizer(lr=lr, wd=wd)
        for e in range(num_epochs):
            self.train()
            epoch_loss = []
            taxi_epoch_loss = []
            ride_epoch_loss = []
            for batch_id, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                taxi_pred, ride_pred = self.forward(x)
                taxi_true, ride_true = torch.chunk(y, 2, dim=-1)

                taxi_loss = self.loss_fun(taxi_pred, taxi_true)
                ride_loss = self.loss_fun(ride_pred, ride_true)

                if self.uncertainty_weighting:
                    loss = (1 / (2 * (self.sigma_1 ** 2))) * taxi_loss + (1 / (2 * (self.sigma_2 ** 2))) * ride_loss \
                           + torch.log(self.sigma_1 * self.sigma_2)
                else:
                    loss = taxi_loss + ride_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())
                taxi_epoch_loss.append(taxi_loss.item())
                ride_epoch_loss.append(ride_loss.item())

            epoch_loss = np.mean(epoch_loss).item()
            taxi_epoch_loss = np.mean(taxi_epoch_loss).item()
            ride_epoch_loss = np.mean(ride_epoch_loss).item()
            self.train_progress.append(epoch_loss)
            print('Train Epoch: %s/%s, Loss: %.4f, TaxiLoss: %.4f, RideLoss: %.4f, time: %.2fs' %
                  (e + 1, num_epochs, epoch_loss, taxi_epoch_loss, ride_epoch_loss, time.time() - start_time))

            val_loss = self.val_exec(val_loader, device)
            self.val_progress.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = e
                self.save_model(e, lr)
            print()

        self.save_progress(num_epochs, lr)
        print('Best Epoch:', best_epoch, 'Val Loss:', best_val_loss)

    def val_exec(self, val_loader, device):
        self.eval()
        with torch.no_grad():
            batch_loss = []
            taxi_batch_loss = []
            ride_batch_loss = []
            for batch_id, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                taxi_pred, ride_pred = self.forward(x)
                taxi_true, ride_true = torch.chunk(y, 2, dim=-1)

                taxi_loss = self.loss_fun(taxi_pred, taxi_true)
                ride_loss = self.loss_fun(ride_pred, ride_true)
                loss = taxi_loss + ride_loss

                batch_loss.append(loss.item())
                taxi_batch_loss.append(taxi_loss.item())
                ride_batch_loss.append(ride_loss.item())

            val_loss = np.mean(batch_loss).item()
            taxi_val_loss = np.mean(taxi_batch_loss).item()
            ride_val_loss = np.mean(ride_batch_loss).item()

            print('VAL Phase: Loss %.4f, TaxiLoss: %.4f, RideLoss: %.4f,' % (val_loss, taxi_val_loss, ride_val_loss))
        return val_loss

    def test_exec(self, test_loader, device):
        print('Testing on', device)
        self.eval()
        taxi_outputs, taxi_targets = [], []
        ride_outputs, ride_targets = [], []
        with torch.no_grad():
            for batch_id, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                taxi_pred, ride_pred = self.forward(x)
                taxi_true, ride_true = torch.chunk(y, 2, dim=-1)
                taxi_outputs.append(taxi_pred)
                taxi_targets.append(taxi_true)
                ride_outputs.append(ride_pred)
                ride_targets.append(ride_true)
        taxi_outputs = torch.cat(taxi_outputs, dim=0)
        taxi_targets = torch.cat(taxi_targets, dim=0)
        ride_outputs = torch.cat(ride_outputs, dim=0)
        ride_targets = torch.cat(ride_targets, dim=0)
        return taxi_outputs, taxi_targets, ride_outputs, ride_targets

    def save_model(self, epoch, lr):
        prefix = 'checkpoints/'
        file_marker = self.model_name + '_lr' + str(lr) + '_e' + str(epoch + 1)
        model_path = time.strftime(prefix + '%m%d_%H_%M_' + file_marker + '.pth')
        torch.save(self.state_dict(), model_path)
        print('save parameters to file: %s' % model_path)

    def load_model(self, model_path, device):
        self.load_state_dict(torch.load(model_path, map_location=device))
        print('load parameters from file: %s' % model_path)

    def save_progress(self, epoch, lr):
        prefix = 'logs/'
        file_marker = self.model_name + '_lr' + str(lr) + '_e' + str(epoch)
        log_path = time.strftime(prefix + '%m%d_%H_%M_' + file_marker + '.npy')
        np.save(log_path, np.array((self.train_progress, self.val_progress)))
        print('save log to file: %s' % log_path)

    def load_progress(self, log_path):
        total_progress = np.load(log_path)
        self.train_progress = total_progress[0]
        self.val_progress = total_progress[1]

    def plot_loss_curve(self):
        fig, ax = plt.subplots()
        x = np.arange(1, len(self.train_progress) + 1)
        ax.plot(x, self.train_progress, label='Training Loss')
        ax.plot(x, self.val_progress, label='Val Loss')
        ax.legend()
        ax.set_xlabel('Num of Epochs')
        ax.set_ylabel('Value of Loss')
        plt.show()
