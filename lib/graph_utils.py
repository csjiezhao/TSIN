import torch
import numpy as np
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt


def cal_adaptive_matrix(emb_s, emb_d):
    """
    :param emb_s: (num_nodes, emb_dim)
    :param emb_d: (num_nodes, emb_dim)
    :return:
    """
    dot_prod = torch.mm(emb_s, emb_d.t())
    adaptive_matrix = torch.softmax(torch.relu(dot_prod), dim=1)
    return adaptive_matrix


def load_default_mat(city, mat_type, device):
    mat_path = 'data/' + city + '/' + mat_type + '_matrix.npy'
    mat = np.load(mat_path)
    return torch.FloatTensor(mat).to(device)


def cal_sym_norm_lap(A):
    """
    Symmetrically normalized Laplacian
    D^{-0.5}(D-A)D^{-0.5} = I-D^{-0.5}(A)D^{-0.5}
    """
    D = torch.diag(A.sum(dim=1))
    D_pow = D ** (-0.5)
    D_pow.masked_fill_(D_pow == float('inf'), 0)
    return torch.eye(A.shape[0], device=A.device) - D_pow.mm(A).mm(D_pow)


def cal_sym_norm_self_mat(A):
    """
    Graph matrix for Standard GCN
    A_tilde = (A + I)
    D_tilde = Degree(A_tilde)
    D_tilde^{-0.5}(A + I)D_tilde^{-0.5}
    """
    A_tilde = A + torch.eye(A.size(0), device=A.device)  # add self-loop
    D_tilde = torch.diag(A_tilde.sum(dim=1))
    D_tilde_pow = D_tilde ** (-0.5)
    D_tilde_pow.masked_fill_(D_tilde_pow == float('inf'), 0)
    return D_tilde_pow.mm(A).mm(D_tilde_pow)


def cal_scaled_norm_lap(A):
    """
    Graph matrix for calculate K-order Laplacian in ChebyshevNet
    """
    L = torch.diag(A.sum(dim=1)) - A
    # L = cal_sym_norm_lap(A)
    evals = torch.eig(L, eigenvectors=True)[0][:, 0]
    lambda_max = torch.max(evals)
    # evals = torch.linalg.eig(L)[0].real
    # lambda_max = torch.max(evals)
    L_tilde = (2 * L) / lambda_max - torch.eye(A.shape[0], device=A.device).float()
    return L_tilde


def cal_cheb_k_lap(A, K):
    """
    Calculate K-order Laplacian in ChebyshevNet
    """
    L_tilde = cal_scaled_norm_lap(A)
    cheb_polynomials = [torch.eye(A.shape[0], device=A.device)]
    if K == 1:
        return cheb_polynomials
    else:
        cheb_polynomials.append(L_tilde)
        if K == 2:
            return cheb_polynomials
        else:
            for i in range(2, K):
                cheb_polynomials.append(2 * torch.mm(L_tilde, cheb_polynomials[i - 1]) - cheb_polynomials[i - 2])
    return cheb_polynomials


def cal_out_diffusion_matrix(A):
    """
    Do^{-1}A = A/row_sum(A)
    """
    Do = torch.diag(A.sum(dim=1))
    Do_inv = Do ** (-1)
    Do_inv.masked_fill_(Do_inv == float('inf'), 0.)
    return torch.matmul(Do_inv, A)


def cal_in_diffusion_matrix(A):
    """
    Di^{-1}A = A/col_sum(A)
    """
    Di = torch.diag(A.sum(dim=0))
    Di_inv = Di ** (-1)
    Di_inv.masked_fill_(Di_inv == float('inf'), 0.)
    return torch.matmul(Di_inv, A)


def cal_cosine_similarity(node_embeddings):
    dot_prod = torch.mm(node_embeddings, node_embeddings.t())
    norm2 = torch.norm(node_embeddings, p=2, dim=1, keepdim=True)
    cosine = dot_prod / (torch.mm(norm2, norm2.t()) + 1e-7)
    return cosine


def cal_asy_cosine_similarity(emb_s, emb_d):
    dot_prod = torch.mm(emb_s, emb_d.t())
    norm_s = torch.norm(emb_s, p=2, dim=1, keepdim=True)
    norm_d = torch.norm(emb_d, p=2, dim=1, keepdim=True)
    cosine = dot_prod / (torch.mm(norm_s, norm_d.t()) + 1e-7)
    return cosine


def cal_dot_prod_similarity(node_embeddings):
    dot_prod = torch.mm(node_embeddings, node_embeddings.t())
    return F.softmax(F.relu(dot_prod), dim=1)


def cal_asy_dot_prod_similarity(emb_s, emb_d):
    dot_prod = torch.mm(emb_s, emb_d.t())
    return F.softmax(F.relu(dot_prod), dim=1)


def cal_1st_proximity(node_embeddings):
    dot_prod = torch.mm(node_embeddings, node_embeddings.t())
    p_1st = torch.sigmoid(dot_prod)
    return p_1st


def cal_asy_1st_proximity(emb_s, emb_d):
    dot_prod = torch.mm(emb_s, emb_d.t())
    p_1st = torch.sigmoid(dot_prod)
    return p_1st


def cal_2nd_proximity(node_embeddings):
    dot_prod = torch.mm(node_embeddings, node_embeddings.t())
    p_2nd = torch.exp(dot_prod) / torch.exp(dot_prod).sum()
    return p_2nd


def cal_asy_2nd_proximity(emb_s, emb_d):
    dot_prod = torch.mm(emb_s, emb_d.t())
    p_2nd = torch.exp(dot_prod) / torch.exp(dot_prod).sum()
    return p_2nd


def cal_k_order(matrix, k):
    matrices = [torch.zeros_like(matrix, device=matrix.device), matrix]
    if k > 1:
        for i in range(2, k + 1):
            matrices.append(2 * torch.matmul(matrix, matrices[-1]) - matrices[-2])
    return matrices


def plot_graph_matrix(matrix, name):
    ax = sns.heatmap(matrix, linewidth=0.01, linecolor='black', cmap='viridis')
    ax.set_title(name)
    plt.savefig('logs/' + name + '.png')
    plt.cla()
    plt.clf()