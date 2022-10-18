import torch


def masked_mse_loss(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)


def masked_mean_absolute_error(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def masked_mean_absolute_percentage_error(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs((true - pred) / true))


def masked_root_mean_squared_error(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((true - pred) ** 2) ** 0.5


def metrics(pred, true, mae_mask=None, mape_mask=0, rmse_mask=None):
    mae = masked_mean_absolute_error(pred, true, mae_mask).item()
    mape = masked_mean_absolute_percentage_error(pred, true, mape_mask).item() * 100
    rmse = masked_root_mean_squared_error(pred, true, rmse_mask).item()
    return round(mae, 4), round(mape, 4), round(rmse, 4)
