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

from carbon_dataset import CarbonDataset_MOMENT_v2, CarbonDataset_BASE_v3

from utils_carbon import re_scale_data_v2, control_randomness
import argparse


def main(args):

    randomseed = 1994

    # code_mode = "train"
    # code_mode = "test"

    # Set random seeds for PyTorch, Numpy etc.
    control_randomness(seed=randomseed)

    # region_list = ["CISO", "PJM", "ERCO", "ISNE", "SE", "DE"]
    # CISO, PJM, ERCOT, ISO-NE, BPAT, FPL, NYISO, SE, DE, PL, ES, NL, AUS-QLD
    # CISO, PJM, ERCO, ISNE, BPAT, FPL, NYISO, SE, DE, PL, ES, NL, AUS_QLD

    # region_list = ["CISO", "PJM", "ERCO", "ISNE", "BPAT", "FPL", "NYISO", "SE", "DE", "PL", "ES", "NL", "AUS_QLD"]

    # region_list = ["BPAT", "FPL", "NYISO", "PL", "ES", "NL", "AUS_QLD"]

    # region_list = ['Argentina', 'Chile', 'India-North', 'Nigeria', 'Taiwan']

    region_list_1 = ["CISO", "PJM", "ERCO", "ISNE", "BPAT", "FPL", "NYISO", "SE", "DE", "PL", "ES", "NL", "AUS_QLD"]

    region_list_2 = ['Argentina', 'Chile', 'India-North', 'India-South', 'New-Zealand', 'Nigeria', 'Taiwan', 'Uruguay']


    data_base_dir_t = "./data"
    # region_t = "BPAT"
    region_t = args.region

    print(f"\n\nTesting region: {region_t}")

    carbon_type_t = "lifecycle"
    input_seq_len_t = 24
    forecast_horizon_t = 96
    data_split_t = "test"
    data_stride_len_t = 24
    extra_f_time = False
    extra_f_weather = False
    # if extra_f_weather:
    #     input_seq_len_t = forecast_horizon_t  # because weather if forecasted, its length is the same as forecast horizon

    # make it always False, cut the extra features when training or testing
    only_carbon_forecast = False

    batch_size = 8

    if region_t in region_list_1:
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
        
    if region_t in region_list_2:

        test_dataset = CarbonDataset_BASE_v3(data_base_dir=data_base_dir_t,
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

    # histories_unscaled = re_scale_data_v2(histories, test_dataset)
    
    # ft_min, ft_max = test_dataset.ft_min, test_dataset.ft_max
    # w_ft_min, w_ft_max = test_dataset.w_ft_min, test_dataset.w_ft_max

    # trues = test_dataset.unscale(trues)
    # preds = test_dataset.unscale(trues)
    # histories = re_scale_data(histories, test_dataset.scaler)

    # trues_unscaled = test_dataset.scaler.inverse_transform(trues)
    # preds_unscaled = test_dataset.scaler.inverse_transform(preds)

    # ForecastingMetrics(mae=98.94003, mse=15254.197, mape=38.385576009750366, smape=33.7039440870285, rmse=123.50788)
    # 'none' | 'mean' | 'sum'.
    metrics = get_forecasting_metrics(y=trues_unscaled, y_hat=preds_unscaled, reduction='mean')

    print(f"Test RMSE: {metrics.rmse:.2f} : Test MAPE: {metrics.mape:.2f}")

    np.save('./temp/data_de_gt.npy', trues_unscaled)
    np.save('./temp/data_de_pre.npy', preds_unscaled)


    print("\n############################ FOUR DAY AVERAGE METRICS #############################")

    from utils_carbon import calcuate_mape
    mean_mape, median_mape, percentile_90th, percentile_95th = calcuate_mape(trues_unscaled, preds_unscaled)

    print("############################ FOUR DAY AVERAGE METRICS #############################\n")

    num_splits = trues_unscaled.shape[1] // 24

    mean_mape_list = []
    median_mape_list = []
    percentile_90th_list = []
    percentile_95th_list = []

    for i in range(num_splits):
        y = trues_unscaled[:, i * 24: (i + 1) * 24]
        y_hat = preds_unscaled[:, i * 24: (i + 1) * 24]

        # metrics = get_forecasting_metrics(y=y, y_hat=y_hat, reduction='mean')
        mean_mape, median_mape, percentile_90th, percentile_95th = calcuate_mape(y, y_hat)

        mean_mape_list.append(mean_mape)
        median_mape_list.append(median_mape)
        percentile_90th_list.append(percentile_90th)
        percentile_95th_list.append(percentile_95th)


    print("\n############################ Daily METRICS #############################\n")

    print("\nMWAN MAPE")
    print(mean_mape_list)

    print("\nMedian MAPE")
    print(median_mape_list)

    print("\nPercentile 90th MAPE")
    print(percentile_90th_list)

    print("\nPercentile 95th MAPE")
    print(percentile_95th_list)


if  __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str, default="moment", help="The type of models we are testing (chronos | moment)"
    )

    parser.add_argument(
        "--region", type=str, default="DE", help="The type of data we are testing (building | electricity | electricity_uci | ecobee)"
    )
    args = parser.parse_args()

    main(args)


