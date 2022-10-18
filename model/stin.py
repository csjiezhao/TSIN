import torch
import torch.nn as nn
from lib.graph_utils import cal_adaptive_matrix

"""
Please note that here we give the basic framework of TSIN, 
the detailed code will be released after the paper is published.
"""

class TSIN(nn.Module):
    def __init__(self, predefined_adj, args):
        super(TSIN, self).__init__()
        self.predefined_mat = predefined_adj
        self.num_nodes = args.get('num_nodes')
        self.emb_dim = args.get('emb_dim')
        self.in_dim = args.get('in_dim')
        self.start_conv_dim = args.get('start_conv_dim')
        self.tem_dim = args.get('tem_dim')
        self.spa_dim = args.get('spa_dim')
        self.spatial_first = args.get('spatial_first')
        self.Kt = args.get('Kt')
        self.dilation = args.get('dilation')
        self.tem_interact = args.get('tem_interact')
        self.space_interact = args.get('spa_interact')

        self.seq_len = args.get('seq_len')
        self.pred_len = args.get('pred_len')

        self.taxi_Es = nn.Parameter(torch.randn(self.num_nodes, self.emb_dim))
        self.taxi_Ed = nn.Parameter(torch.randn(self.num_nodes, self.emb_dim))
        self.ride_Es = nn.Parameter(torch.randn(self.num_nodes, self.emb_dim))
        self.ride_Ed = nn.Parameter(torch.randn(self.num_nodes, self.emb_dim))

        if self.spatial_first:
            self.head_dim = self.spa_dim
            self.tail_dim = self.tem_dim
        else:
            self.head_dim = self.tem_dim
            self.tail_dim = self.spa_dim

        self.num_blocks = len(self.Kt)
        self.block_lens = [self.seq_len]
        [self.block_lens.append(self.block_lens[-1] - (self.Kt[i] - 1) * self.dilation[i])
         for i in range(self.num_blocks)]

        print(self.num_blocks, 'Blocks :', self.block_lens)

        self.blocks = nn.ModuleList(
            [STBlock(self.in_dim, self.tem_dim, self.spa_dim, self.Kt[0], self.dilation[0],
                     self.block_lens[0], self.spatial_first, self.tem_interact[0], self.space_interact[0])])
        self.blocks.extend([STBlock(self.tail_dim, self.tem_dim, self.spa_dim, self.Kt[i], self.dilation[i],
                                    self.block_lens[i], self.spatial_first, self.tem_interact[i], self.space_interact[i])
                            for i in range(1, self.num_blocks)])

        self.taxi_fuse_layer = nn.Parameter(torch.randn(self.num_blocks * self.tail_dim, 8 * self.tail_dim))
        self.ride_fuse_layer = nn.Parameter(torch.randn(self.num_blocks * self.tail_dim, 8 * self.tail_dim))

        self.taxi_output_layer = nn.Parameter(torch.randn(8 * self.tail_dim, self.pred_len))
        self.ride_output_layer = nn.Parameter(torch.randn(8 * self.tail_dim, self.pred_len))

    def forward(self, x):
        """
        :param x: (batch, seq_len, num_nodes, in_dim * 2)
        :return:
        """
        # taxi to taxi
        t2t_mat = cal_adaptive_matrix(self.taxi_Es, self.taxi_Ed)
        # ride to taxi
        r2t_mat = cal_adaptive_matrix(self.ride_Es, self.taxi_Ed)
        # ride to ride
        r2r_mat = cal_adaptive_matrix(self.ride_Es, self.ride_Ed)
        # taxi to ride
        t2r_mat = cal_adaptive_matrix(self.taxi_Es, self.ride_Ed)

        x = x.permute(0, 3, 1, 2)  # (batch_size, in_dim, seq_len, num_nodes)
        taxi_x, ride_x = torch.chunk(x, 2, dim=1)

        taxi_skip_connections = []
        ride_skip_connections = []
        for i in range(self.num_blocks):
            taxi_x, ride_x = self.blocks[i](taxi_x, t2t_mat, r2t_mat, ride_x, r2r_mat, t2r_mat)
            taxi_skip = taxi_x[:, :, -1, :]
            ride_skip = ride_x[:, :, -1, :]

            taxi_skip_connections.append(taxi_skip)  # (B, F, T, N)
            ride_skip_connections.append(ride_skip)

        taxi_skip_feats = torch.cat(taxi_skip_connections, dim=1).permute(0, 2, 1)
        ride_skip_feats = torch.cat(ride_skip_connections, dim=1).permute(0, 2, 1)

        taxi_x = torch.relu(taxi_skip_feats.matmul(torch.relu(self.taxi_fuse_layer)))
        ride_x = torch.relu(ride_skip_feats.matmul(torch.relu(self.ride_fuse_layer)))

        taxi_x = torch.matmul(taxi_x, self.taxi_output_layer).unsqueeze(1)
        ride_x = torch.matmul(ride_x, self.ride_output_layer).unsqueeze(1)

        return taxi_x, ride_x
