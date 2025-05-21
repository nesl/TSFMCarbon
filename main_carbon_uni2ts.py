import argparse
import pandas as pd  # requires: pip install pandas
import pdb 
from utils_carbon import plot_pred, signal_simulate, test_foundation_model, \
    load_foundation_model, save_results_for_model

# from momentfm.utils.utils import control_randomness

from get_data_carbon import get_carbon_data, get_carbon_batch_data


from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from datasets import load_dataset

import numpy as np

from forecasting_metrics import get_forecasting_metrics

from carbon_dataset import CarbonDataset_CHRONOS
from carbon_dataset import CarbonDataset_MOMENT_v2, CarbonDataset_BASE_v3

from torch.utils.data import DataLoader
import torch       

from utils_carbon import re_scale_data_v2, control_randomness

from models import Uni2TSModel


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = 1e-7  # prevent division by zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def main(args):

    randomseed = 1994

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
    data_stride_len_t = 1
    extra_f_time = False
    extra_f_weather = False
    # if extra_f_weather:
    #     input_seq_len_t = forecast_horizon_t  # because weather if forecasted, its length is the same as forecast horizon

    # make it always False, cut the extra features when training or testing
    only_carbon_forecast = False

    batch_size = 1

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

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # model = load_foundation_model(args, input_seq_len_t, forecast_horizon_t)

    # for split data, as in ViT
    # pacth_len_t = 2
    model = Uni2TSModel(prediction_length=forecast_horizon_t, 
                        context_length=input_seq_len_t, 
                        size="large", patch_size="auto", device="cuda")

    # model.init()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    
    trues, preds, histories, losses = [], [], [], []

    test_iteration = 2
    iteration = 0
    with torch.no_grad():

        # input_mask only use for moment
        for timeseries, forecast, input_mask in tqdm(test_loader, total=len(test_loader)):

            timeseries = timeseries.float()  # torch.Size([1, 13, 24])
            input_mask = input_mask   # torch.Size([1, 512])
            forecast = forecast.float()  # torch.Size([1, 13, 96])

            # historical_data, gt, result_preds = None, None, None
            # if args.model == "chronos":
            # chronos only support univariate time series
            historical_data = timeseries[:, 0, :] # torch.Size([1, 24])
            historical_data = historical_data.squeeze()[: input_seq_len_t]   # torch.Size([24])

            result_preds = model(historical_data)  # (96,)
            # # result = test_foundation_model(args, model, sampling_rate, pred_hrz, data, test_data)
            # result_preds = test_foundation_model(args, model, forecast_horizon_t, historical_data)  # (96,)
            result_preds = np.expand_dims(np.expand_dims(result_preds, axis=0), axis=0)  # (1, 1, 96)

            gt = forecast[:, 0, :]  # torch.Size([8, 96])
            gt = gt.squeeze().cpu().numpy()
            gt = np.expand_dims(np.expand_dims(gt, axis=0), axis=0)  # (1, 1, 96)

            trues.append(gt)
            preds.append(result_preds)
            # histories.append(data)

            iteration += 1

            # if iteration > test_iteration:
            #     break

    trues = np.concatenate(trues, axis=0)  # (X, 1, 96)
    trues_unscaled = re_scale_data_v2(trues, test_dataset)

    preds = np.concatenate(preds, axis=0)  # (X, 1, 96)
    preds_unscaled = re_scale_data_v2(preds, test_dataset)

    # histories = np.concatenate(histories, axis=0) # (X, 1, 512)
    # metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction='mean')
    metrics = get_forecasting_metrics(y=trues_unscaled, y_hat=preds_unscaled, reduction='mean')


    # print(f"Epoch 0: Test MSE: {metrics.mse:.3f} | Test MAE: {metrics.mae:.3f} : Test MAPE: {metrics.mape:.3f}")
    # print(f"Test RMSE: {metrics.rmse:.5f} : Test MAPE: {metrics.mape:.5f}")
    print(f"Test RMSE: {metrics.rmse:.2f} : Test MAPE: {metrics.mape:.2f}")

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
        "--model", type=str, default="uni2ts", help="The type of models we are testing (chronos | moment)"
    )

    parser.add_argument(
        "--region", type=str, default="Argentina", help="The type of data we are testing (building | electricity | electricity_uci | ecobee)"
    )
    args = parser.parse_args()

    main(args)



