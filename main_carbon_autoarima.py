import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from pmdarima import auto_arima

from momentfm.utils.forecasting_metrics import get_forecasting_metrics
from carbon_dataset import CarbonDataset_MOMENT_v2, CarbonDataset_BASE_v3
from utils_carbon import re_scale_data_v2, control_randomness

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
    data_stride_len_t = 1
    extra_f_time = False
    extra_f_weather = False
    # if extra_f_weather:
    #     input_seq_len_t = forecast_horizon_t  # because weather if forecasted, its length is the same as forecast horizon

    # make it always False, cut the extra features when training or testing
    only_carbon_forecast = False

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

    all_preds = []
    all_trues = []

    # test_num = 5
    print("Running AutoARIMA forecasting per test sample...")
    for i in tqdm(range(len(test_dataset))):
        x_test, y_test, _ = test_dataset[i]

        # Use only carbon intensity (first channel)
        x_input = x_test[0][:input_seq_len_t]    # (input_seq_len,)
        y_target = y_test[0]   # (forecast_horizon,)

        # Fit AutoARIMA on test input sequence
        model = auto_arima(
            x_input,
            seasonal=False,
            suppress_warnings=True,
            error_action='ignore'
        )

        y_pred = model.predict(n_periods=forecast_horizon_t)

        all_trues.append(y_target[np.newaxis, :])  # shape (1, 96)
        all_preds.append(y_pred[np.newaxis, :])    # shape (1, 96)

        # if i > test_num:
        #     break

    # Stack predictions and targets: shape (N, 1, 96)
    all_trues = np.stack(all_trues, axis=0)  # (N, 1, 96)
    all_preds = np.stack(all_preds, axis=0)  # (N, 1, 96)

    # Unscale
    trues_unscaled = re_scale_data_v2(all_trues, test_dataset)
    preds_unscaled = re_scale_data_v2(all_preds, test_dataset)

    # Evaluate
    metrics = get_forecasting_metrics(y=trues_unscaled, y_hat=preds_unscaled, reduction='mean')
    print(f"[AutoARIMA] Test RMSE: {metrics.rmse:.2f} | Test MAPE: {metrics.mape:.2f}")


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
        "--model", type=str, default="autoarima", help="The type of models we are testing (chronos | moment)"
    )

    parser.add_argument(
        "--region", type=str, default="CISO", help="The type of data we are testing (building | electricity | electricity_uci | ecobee)"
    )
    args = parser.parse_args()

    main(args)

