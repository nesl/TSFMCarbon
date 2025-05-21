import numpy as np
import torch
import torch.cuda.amp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# from momentfm.utils.utils import control_randomness
from momentfm.data.informer_dataset import InformerDataset
# from momentfm.data.carbon_dataset import CarbonDataset

from momentfm.utils.forecasting_metrics import get_forecasting_metrics

from momentfm import MOMENTPipeline
from pprint import pprint
import torch
import matplotlib.pyplot as plt
import os

from carbon_dataset import CarbonDataset_MOMENT, CarbonDataset_MOMENT_v2

from utils_carbon import re_scale_data_v2, control_randomness


if  __name__ == '__main__':
    randomseed = 1994

    # code_mode = "train"
    # code_mode = "test"

    # Set random seeds for PyTorch, Numpy etc.
    control_randomness(seed=randomseed)
    region_list = ["CISO", "PJM", "ERCO", "ISNE", "SE", "DE"]


    data_base_dir_t = "./data"
    region_t = "PJM"
    carbon_type_t = "lifecycle"
    input_seq_len_t = int(1 * 24)
    forecast_horizon_t = int(30 * 24)
    data_split_t = "test"
    data_stride_len_t = 1
    extra_f_time = False
    extra_f_weather = True
    # if extra_f_weather:
    #     input_seq_len_t = forecast_horizon_t  # because weather if forecasted, its length is the same as forecast horizon

    # make it always False, cut the extra features when training or testing
    only_carbon_forecast = False

    batch_size = 8

    test_dataset = CarbonDataset_MOMENT_v2(data_base_dir=data_base_dir_t,
                                 region=region_t,
                                 carbon_type=carbon_type_t,
                                 input_seq_len=input_seq_len_t,
                                 forecast_horizon=forecast_horizon_t,
                                 data_split=data_split_t,
                                 data_stride_len=data_stride_len_t, 
                                 extra_features_time=extra_f_time, 
                                 extra_features_weather=extra_f_weather, 
                                 only_carbon_forecast=only_carbon_forecast)

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={
            'task_name': 'forecasting',
            'forecast_horizon': forecast_horizon_t,
            'head_dropout': 0.1,
            'weight_decay': 0,
            'freeze_encoder': True,   # Freeze the patch embedding layer
            'freeze_embedder': True,  # Freeze the transformer encoder
            'freeze_head': False,     # The linear forecasting head must be trained
        },
        # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
        )

    model.init()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = torch.nn.MSELoss()
    criterion = criterion.to(device)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    cur_epoch = 0

    model.eval()
    trues, preds, histories, losses = [], [], [], []
    with torch.no_grad():
        for timeseries, forecast, input_mask in tqdm(test_loader, total=len(test_loader)):

            timeseries = timeseries.float().to(device)  # torch.Size([8, 13, 512])
            input_mask = input_mask.to(device)   # torch.Size([8, 512])
            forecast = forecast.float().to(device)  # torch.Size([8, 13, 96])

            with torch.amp.autocast(device_type='cuda'):
            # with torch.cuda.amp.autocast():
                output = model(x_enc=timeseries, input_mask=input_mask)
            
            loss = criterion(output.forecast, forecast)  # output.forecast:  torch.Size([8, 13, 96])    
            # losses.append(loss.item())

            trues.append(forecast.detach().cpu().numpy())         # (8, 13, 96)
            preds.append(output.forecast.detach().cpu().numpy())  # (8, 13, 96)
            histories.append(timeseries.detach().cpu().numpy())   # (8, 13, 512)
    
    # losses = np.array(losses)
    # average_loss = np.average(losses)

    trues = np.concatenate(trues, axis=0)  # (5185, 13, 96)
    preds = np.concatenate(preds, axis=0)  # (5185, 13, 96)
    histories = np.concatenate(histories, axis=0) # (5185, 13, 512)
    # tures shape: (5185, 13, 96), preds shape: (5185, 13, 96), histories shape: (5185, 13, 512)
    print(f"\ntures shape: {trues.shape}, preds shape: {preds.shape}, histories shape: {histories.shape}\n")

    trues = trues[:, 0:1, :]  # shape will be (5185, 1, 96), only take the first feature of carbon intensity
    trues_unscaled = re_scale_data_v2(trues, test_dataset)

    preds = preds[:, 0:1, :]  # shape will be (5185, 1, 96), only take the first feature of carbon intensity
    preds_unscaled = re_scale_data_v2(preds, test_dataset)
    
    # trues: (4561, 720)
    # preds: (4561, 720)

    metrics = get_forecasting_metrics(y=trues_unscaled, y_hat=preds_unscaled, reduction='mean')


    num_splits = trues_unscaled.shape[1] // 24
    metrics_list = []

    for i in range(num_splits):
        y = trues_unscaled[:, i * 24: (i + 1) * 24]
        y_hat = preds_unscaled[:, i * 24: (i + 1) * 24]
        metrics = get_forecasting_metrics(y=y, y_hat=y_hat, reduction='mean')

        metrics_list.append(metrics)

    # print(f"Test RMSE: {metrics.rmse:.2f} : Test MAPE: {metrics.mape:.2f}")
    metric_row = "& " + " & ".join(f"{m.mape:.2f}" for m in metrics_list) + " \\\\"
    print(metric_row)

    # metric_row = metric_row+ " & " + f"{metrics.mape:.2f}" + " \\\\"

    # print(metric_row)



