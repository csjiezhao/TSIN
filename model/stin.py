import torch
import torch.nn as nn
from lib.graph_utils import cal_adaptive_matrix


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

class STBlock(nn.Module):
    def __init__(self, in_dim, tem_dim, spa_dim, Kt, dilation, cur_len, spatial_first, tem_interact, spa_interact):
        super(STBlock, self).__init__()
        self.spatial_first = spatial_first
        if self.spatial_first:
            self.co_spa_conv = CoSpatialConv(interact=spa_interact, seq_len=cur_len,
                                             s_in_dim=in_dim, s_out_dim=spa_dim)
            self.co_tem_conv = CoTemporalConv(interact=tem_interact, t_in_dim=spa_dim,
                                              t_out_dim=tem_dim, Kt=Kt, dilation=dilation)
            self.taxi_align_conv = ResAlign(in_dim, tem_dim)
            self.ride_align_conv = ResAlign(in_dim, tem_dim)
            self.taxi_batch_norm = nn.BatchNorm2d(tem_dim)
            self.ride_batch_norm = nn.BatchNorm2d(tem_dim)
        else:
            self.co_tem_conv = CoTemporalConv(interact=tem_interact, t_in_dim=in_dim,
                                              t_out_dim=tem_dim, Kt=Kt, dilation=dilation)
            self.co_spa_conv = CoSpatialConv(interact=spa_interact, seq_len=cur_len - (Kt - 1) * dilation,
                                             s_in_dim=tem_dim, s_out_dim=spa_dim)
            self.taxi_align_conv = ResAlign(in_dim, spa_dim)
            self.ride_align_conv = ResAlign(in_dim, spa_dim)
            self.taxi_batch_norm = nn.BatchNorm2d(spa_dim)
            self.ride_batch_norm = nn.BatchNorm2d(spa_dim)

        self.align_len = cur_len - (Kt - 1) * dilation

    def forward(self, taxi_x, t2t_mat, r2t_mat, ride_x, r2r_mat, t2r_mat):
        """
        :param taxi_x: (B, in_dim, T, N)
        :param t2t_mat: (N, N) taxi to taxi
        :param r2t_mat: (N, N) ride to taxi
        :param ride_x: (B, in_dim, T, N)
        :param r2r_mat: (N, N) ride to ride
        :param t2r_mat: (N, N) taxi to ride
        :return:
        """
        taxi_shortcut = self.taxi_align_conv(taxi_x[:, :, -self.align_len:, :])
        ride_shortcut = self.ride_align_conv(ride_x[:, :, -self.align_len:, :])
        if self.spatial_first:
            taxi_x, ride_x = self.co_spa_conv(taxi_x, t2t_mat, r2t_mat, ride_x, r2r_mat, t2r_mat)
            taxi_x, ride_x = self.co_tem_conv(taxi_x, ride_x)
        else:
            taxi_x, ride_x = self.co_tem_conv(taxi_x, ride_x)
            taxi_x, ride_x = self.co_spa_conv(taxi_x, t2t_mat, r2t_mat, ride_x, r2r_mat, t2r_mat)
        taxi_x = self.taxi_batch_norm(taxi_shortcut + taxi_x)
        ride_x = self.ride_batch_norm(ride_shortcut + ride_x)
        return taxi_x, ride_x


class CoSpatialConv(nn.Module):
    def __init__(self, interact, seq_len, s_in_dim, s_out_dim):
        super(CoSpatialConv, self).__init__()
        self.taxi_spa_conv = HetGraphConv(interact, seq_len, s_in_dim, s_out_dim)
        self.ride_spa_conv = HetGraphConv(interact, seq_len, s_in_dim, s_out_dim)

    def forward(self, taxi_x, t2t_mat, r2t_mat, ride_x, r2r_mat, t2r_mat):
        """
        :param taxi_x: (B, in_dim, T, N)
        :param t2t_mat: (N, N) taxi to taxi
        :param r2t_mat: (N, N) ride to taxi
        :param ride_x: (B, in_dim, T, N)
        :param r2r_mat: (N, N) ride to ride
        :param t2r_mat: (N, N) taxi to ride
        :return:
        """
        taxi_out = self.taxi_spa_conv(hom_x=taxi_x, hom_mat=t2t_mat, het_x=ride_x, het_mat=r2t_mat)
        ride_out = self.ride_spa_conv(hom_x=ride_x, hom_mat=r2r_mat, het_x=taxi_x, het_mat=t2r_mat)
        return taxi_out, ride_out


class HetGraphConv(nn.Module):
    def __init__(self, interact, cur_len, s_in_dim, s_out_dim,):
        super(HetGraphConv, self).__init__()
        self.cur_len = cur_len
        self.s_in_dim = s_in_dim
        self.s_out_dim = s_out_dim
        self.interact = interact

        if interact:
            self.W = nn.Conv2d(in_channels=3 * self.s_in_dim, out_channels=2 * self.s_out_dim, kernel_size=(1, 1),
                               padding=(0, 0), stride=(1, 1), bias=True)
        else:
            self.W = nn.Conv2d(in_channels=2 * self.s_in_dim, out_channels=2 * self.s_out_dim, kernel_size=(1, 1),
                               padding=(0, 0), stride=(1, 1), bias=True)

        self.drop_out = nn.Dropout(p=0.3)

    def forward(self, hom_x, hom_mat, het_x, het_mat):
        """
        :param hom_x: (batch, in_dim, cur_len, num_nodes)
        :param hom_mat: (num_nodes, num_nodes)
        :param het_x: (batch, in_dim, cur_len, num_nodes)
        :param het_mat: (num_nodes, num_nodes)
        :return: (batch, out_dim, cur_len, num_nodes)
        """

        hom_conv = torch.einsum('bcln, nm -> bclm', (hom_x, hom_mat))

        if self.interact:
            het_conv = torch.einsum('bcln, nm -> bclm', (het_x, het_mat))
            out = torch.cat([hom_x, hom_conv, het_conv], dim=1)  # (batch_size, 3*in_dim, seq_len, num_nodes)
        else:
            out = torch.cat([hom_x, hom_conv], dim=1)

        out = self.W(out)  # (batch_size, 2 * out_dim, seq_len, num_nodes)
        out_p, out_q = torch.chunk(out, chunks=2, dim=1)
        out = out_p * torch.sigmoid(out_q)  # (batch_size, out_dim, seq_len, num_nodes)
        out = self.drop_out(out)
        return out


class CoTemporalConv(nn.Module):
    def __init__(self, interact, t_in_dim, t_out_dim, Kt, dilation, Ks=1):
        super(CoTemporalConv, self).__init__()
        self.interact = interact
        self.Kt = Kt
        self.dilation = dilation

        self.taxi_conv2d = nn.Conv2d(t_in_dim, 2 * t_out_dim, kernel_size=(Kt, Ks),
                                     padding=((Kt - 1) * dilation, 0), dilation=dilation)

        self.ride_conv2d = nn.Conv2d(t_in_dim, 2 * t_out_dim, kernel_size=(Kt, Ks),
                                     padding=((Kt - 1) * dilation, 0), dilation=dilation)
        if self.interact:
            self.taxi_het_conv2d = nn.Conv2d(t_in_dim, 2 * t_out_dim, kernel_size=(Kt, Ks))
            self.ride_het_conv2d = nn.Conv2d(t_in_dim, 2 * t_out_dim, kernel_size=(Kt, Ks))

    def forward(self, taxi_x, ride_x):
        """
        :param taxi_x: (batch, t_in_dim, seq_len, num_nodes)
        :param ride_x: (batch, t_in_dim, seq_len, num_nodes)
        """
        if self.interact:
            taxi_het_x = self.taxi_het_conv2d(ride_x[:, :, -self.Kt:, :])
            ride_het_x = self.ride_het_conv2d(taxi_x[:, :, -self.Kt:, :])

            # Self Gated
            taxi_het_x = torch.relu(taxi_het_x)
            ride_het_x = torch.relu(ride_het_x)

            taxi_x = self.taxi_conv2d(taxi_x)
            taxi_x = taxi_x[:, :, (self.Kt - 1) * self.dilation:-(self.Kt - 1) * self.dilation, :]
            taxi_x[:, :, -1:, :] += taxi_het_x

            ride_x = self.ride_conv2d(ride_x)
            ride_x = ride_x[:, :, (self.Kt - 1) * self.dilation:-(self.Kt - 1) * self.dilation, :]
            ride_x[:, :, -1:, :] += ride_het_x
        else:
            taxi_x = self.taxi_conv2d(taxi_x)
            taxi_x = taxi_x[:, :, (self.Kt - 1) * self.dilation:-(self.Kt - 1) * self.dilation, :]

            ride_x = self.ride_conv2d(ride_x)
            ride_x = ride_x[:, :, (self.Kt - 1) * self.dilation:-(self.Kt - 1) * self.dilation, :]

        taxi_x_p, taxi_x_q = torch.chunk(taxi_x, 2, dim=1)
        ride_x_p, ride_x_q = torch.chunk(ride_x, 2, dim=1)
        return taxi_x_p * torch.sigmoid(taxi_x_q), ride_x_p * torch.sigmoid(ride_x_q)


class ResAlign(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResAlign, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduce_conv = nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1))

    def forward(self, x):
        """
        Align the feature dimension
        :param x: (batch, in_dim, seq_len-(Kt-1), num_nodes)
        :return: (batch, out_dim, seq_len-(Kt-1), num_nodes)
        """
        if self.in_dim > self.out_dim:
            x = self.reduce_conv(x)
        elif self.in_dim < self.out_dim:
            batch, _, seq_len, num_nodes = x.shape
            x = torch.cat([x, torch.zeros([batch, self.out_dim - self.in_dim, seq_len, num_nodes],
                                          device=x.device)], dim=1)
        else:
            x = x
        return x
