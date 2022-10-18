num_feats = 1
seq_len = 12
pred_len = 1

chicago_config = {
    'tsin': {'num_nodes': 77, 'emb_dim': 10, 'in_dim': 1, 'tem_dim': 32, 'spa_dim': 32,
             'Kt': [3, 2, 2, 2], 'dilation': [1, 2, 3, 4], 'tem_interact': [False, False, True, True],
             'spa_interact': [True, True, True, True],
             'spatial_first': False, 'seq_len': seq_len, 'num_matrices': 2, 'pred_len': pred_len}
}

nyc_config = {
    'tsin': {'num_nodes': 63, 'emb_dim': 10, 'in_dim': 1, 'tem_dim': 32, 'spa_dim': 32,
             'Kt': [3, 2, 2, 2], 'dilation': [1, 2, 3, 4], 'tem_interact': [False, False, True, True],
             'spa_interact': [True, True, True, True],
             'spatial_first': False, 'seq_len': seq_len, 'num_matrices': 2, 'pred_len': pred_len}
}
