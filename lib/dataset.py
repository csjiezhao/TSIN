import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    def __init__(self, city, interval, seq_step, pred_step, mode):

        if city not in ['chicago', 'nyc']:
            raise ValueError('Unknown City!')

        if interval not in [15, 30, 45, 60]:
            raise ValueError('Time Interval must be one of {15, 30, 45, 60}(min)!')

        print('************************************************************')
        print('City:', city, '; ', 'Interval:', str(interval) + ' min')

        if city == 'chicago':
            taxi_picks = np.load('data/chicago/2019_chi_taxi_pickups_15min.npz')['arr_0'][:, :, np.newaxis]
            ride_picks = np.load('data/chicago/2019_chi_tnp_pickups_15min.npz')['arr_0'][:, :, np.newaxis]
        elif city == 'nyc':
            taxi_picks = np.load('data/nyc/2018_nyc_yellow_pickups_15min.npz')['data'][:, :, np.newaxis] + \
                         np.load('data/nyc/2018_nyc_green_pickups_15min.npz')['data'][:, :, np.newaxis]
            ride_picks = np.load('data/nyc/2018_nyc_fhv_pickups_15min.npz')['data'][:, :, np.newaxis]
        else:
            taxi_picks = None
            ride_picks = None

        data = np.concatenate((taxi_picks, ride_picks), axis=-1)
        data = granularity_transform(data, source=15, target=interval)

        self.X, self.Y, = make_dataset(graph_feats=data, seq_step=seq_step,
                                       pred_step=pred_step, mode=mode, normalize=True)

        assert self.X.shape[0] == self.Y.shape[0], 'Data Error!'
        print('X:', self.X.shape, 'Y:', self.Y.shape)
        print(mode + ' dataset created !')
        print('************************************************************')

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.Y)


def granularity_transform(data_seq, source, target):
    """
    :param data_seq: (num_timestamps, num_nodes, num_feats)
    :param source: source granularity of data_seq (min)
    :param target: target granularity (min)
    :return:
    """
    if source == target:
        return data_seq
    else:
        origin_len = len(data_seq)
        merge_len = int(target / source)
        coarsened_size = int(origin_len / merge_len)
        coarsened_seq = []
        for t in range(coarsened_size):
            coarsened_seq.append(sum(data_seq[t * merge_len: (t + 1) * merge_len]))
        return np.array(coarsened_seq)


def get_samples(data, seq_step, pred_step):
    samples = []
    len_data = data.shape[0]
    for idx in range(len_data):
        if seq_step <= idx and idx + pred_step <= len_data:
            sample_x = data[idx - seq_step: idx, :, :]
            sample_y = data[idx: idx + pred_step, :, :]
            samples.append((sample_x, sample_y))
    samples = [np.stack(i, axis=0) for i in zip(*samples)]
    return samples


def make_dataset(graph_feats, seq_step, pred_step, mode, normalize=True):
    num_timestamps, num_nodes, num_feats = graph_feats.shape

    '''(num_samples, pred_step, num_nodes, num_feats)
       (num_samples, pred_step, num_nodes, num_feats)'''
    samples_x, samples_y = get_samples(graph_feats, seq_step, pred_step)

    '''split points'''
    sp1 = int(num_timestamps * 0.7)
    sp2 = int(num_timestamps * 0.8)

    train_x, val_x, test_x = samples_x[:sp1], samples_x[sp1:sp2], samples_x[sp2:]
    train_y, val_y, test_y = samples_y[:sp1], samples_y[sp1:sp2], samples_y[sp2:]

    if normalize:
        mean = train_x.mean(axis=(0, 1), keepdims=True)
        std = train_x.std(axis=(0, 1), keepdims=True)

        def z_score(x):
            return (x - mean) / std

        train_x = z_score(train_x)
        val_x = z_score(val_x)
        test_x = z_score(test_x)

    if mode == 'train':
        return torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()
    elif mode == 'val':
        return torch.from_numpy(val_x).float(), torch.from_numpy(val_y).float()
    elif mode == 'test':
        return torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float()
    else:
        raise ValueError('Invalid Type of Dataset!')
