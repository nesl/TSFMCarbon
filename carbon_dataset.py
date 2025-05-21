from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

from get_data_carbon import initialize, splitDataset, splitWeatherDataset_v2, fillMissingData, scaleDataset, manipulateTrainingDataShape_v2, inverseDataScaling


from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from sklearn.preprocessing import StandardScaler


class CarbonDataset:
    def __init__(
        self,
        input_seq_len: int = 512,
        forecast_horizon: Optional[int] = 192,
        data_split: str = "train",
        data_stride_len: int = 1,
        task_name: str = "forecasting",
        random_seed: int = 42,
        data_path: str = "../data/ETTh1.csv",
    ):
        """
        Parameters
        ----------
        forecast_horizon : int
            Length of the prediction sequence.
        data_split : str
            Split of the dataset, 'train' or 'test'.
        data_stride_len : int
            Stride length when generating consecutive
            time series windows.
        task_name : str
            The task that the dataset is used for. One of
            'forecasting', or  'imputation'.
        random_seed : int
            Random seed for reproducibility.
        """
        self.predifined_input_length = 512

        self.seq_len = input_seq_len
        self.forecast_horizon = forecast_horizon
        self.full_file_path_and_name = data_path
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.random_seed = random_seed

        # Read data
        self._read_data()

    def _get_borders(self):
        total_len = self.length_timeseries_original
        train_pct, val_pct, test_pct = 0.7, 0.0, 0.30

        n_train = int(total_len * train_pct)
        n_val = int(total_len * val_pct)
        n_test = total_len - n_train - n_val

        # n_train = 12 * 30 * 24
        # n_val = 4 * 30 * 24
        # n_test = 4 * 30 * 24

        train_end = n_train
        val_end = n_train + n_val

        test_start = val_end - self.seq_len # # allow window to start earlier
        test_end = test_start + n_test + self.seq_len

        train = slice(0, train_end)
        test = slice(test_start, test_end)

        return train, test

    def _read_data(self):
        self.scaler = StandardScaler()
        # df = pd.read_csv(self.full_file_path_and_name)
        df = pd.read_csv(self.full_file_path_and_name, delimiter=',', header=0)  # Tells Pandas to use the first row (row 0) of the file as the column names.

        column_name = "carbon_intensity"
        df.rename(columns={df.columns[1]: 'TS'}, inplace=True) # rename the first column of a DataFrame (df) to 'TS'
        df = df[['TS', column_name]] # selecting just two columns from the DataFrame df
        
        df[column_name] = df[column_name].astype(float)
        df['TS'] = pd.to_datetime(df['TS'])     # Converts the values to Pandas datetime objects

        self.length_timeseries_original = df.shape[0]
        self.n_channels = df.shape[1] - 1

        df.drop(columns=["TS"], inplace=True)  # This line removes the "date" column from the DataFrame df.
        df = df.infer_objects(copy=False).interpolate(method="cubic")

        data_splits = self._get_borders()

        train_data = df[data_splits[0]]
        self.scaler.fit(train_data.values)
        df = self.scaler.transform(df.values)

        if self.data_split == "train":
            self.data = df[data_splits[0], :]
        elif self.data_split == "test":
            self.data = df[data_splits[1], :]

        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):

        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len

        pred_end = seq_end + self.forecast_horizon

        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_end = seq_end - self.forecast_horizon
            seq_start = seq_end - self.seq_len

        timeseries = self.data[seq_start:seq_end, :]
        forecast = self.data[seq_end:pred_end, :]

        # if self.seq_len <= self.predifined_input_length:

        pad_len = 512 - self.seq_len
        padding = np.zeros((pad_len, timeseries.shape[1]))
        timeseries = np.vstack((timeseries, padding))  # shape becomes [512, features]

        input_mask = np.concatenate((np.ones(self.seq_len), np.zeros(pad_len)))

        # else:
        #     input_mask = np.ones(self.seq_len)  # (512,))

        timeseries = timeseries.T  # (7, 512)
        forecast = forecast.T     # (7, 192)

        return timeseries, forecast, input_mask
    

    def __len__(self):

        # if self.task_name == "imputation":
        #     return (self.length_timeseries - self.seq_len) // self.data_stride_len + 1
        
        # elif self.task_name == "forecasting":

        # if self.data_split == "train":
        #     return (self.length_timeseries - self.seq_len) // self.data_stride_len + 1
        # elif self.data_split == "test":
        #     return (self.length_timeseries - self.seq_len) // self.data_stride_len + 1
        return (self.length_timeseries - self.seq_len - self.forecast_horizon) // self.data_stride_len + 1


class CarbonDataset_BASE:
    def __init__(self,
                 data_base_dir: str = "/kang/carbon/CarbonCast_main/data",
                 region: str = "CISO",
                 carbon_type: str = "lifecycle",
                 input_seq_len: int = 512,
                 forecast_horizon: int = 192,
                 data_split: str = "train",
                 data_stride_len: int = 1):

        start_col = 1   # index in inFileName
        number_of_features = 6  # carbon_intensity, plus 5 time-related features: (hour_sin, hour_cos, month_sin, month_cos, weekend)
        num_weather_features = 7  # in carboncast the valuse is 12, since it contains source prediction electricty from first tier model
        carbon_intensity_index = 0  # carbon_intensity index in dataset

        self.start_col = start_col
        self.number_of_features = number_of_features
        self.num_weather_features = num_weather_features
        self.carbon_intensity_index = carbon_intensity_index
        
        data_base_dir_region = os.path.join(data_base_dir, region)

        # '/kang/carbon/CarbonCast_main/data/CISO/CISO_direct_emissions.csv'
        in_file_name = os.path.join(data_base_dir_region, f'{region}_{carbon_type}_emissions.csv')
        self.in_file_name = in_file_name

        weather_file_name = f"/mnt/ssd2/kangyang/carbon/TSFM_Building_carbon/data/weather/{region}_weather_2020_2021.csv"
        self.weather_file_name = weather_file_name

        # self.predifined_input_length = 512

        self.seq_len = input_seq_len
        self.forecast_horizon = forecast_horizon
        # self.full_file_path_and_name = data_path
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        # self.random_seed = random_seed

        training_ratio=0.7
        # Read data
        self._read_data(training_ratio=training_ratio)


    def _read_data(self, training_ratio=0.7):

        dataset, weatherDataset, dateTime = initialize(self.in_file_name, self.weather_file_name, self.start_col)

        dataset_time_step, feature_len = dataset.shape
        self.length_timeseries_original = dataset_time_step

        num_days = dataset_time_step // 24
        # training_ratio = 0.7
        # testing_ratio = 1 - training_ratio

        num_train_days = int(num_days * training_ratio)
        num_test_days = num_days - num_train_days  # 220

        # split into train and test
        # print("Spliting dataset into train/test...")
        # trainData: (12264, 6), testData: (5280, 6)
        trainData, testData = splitDataset(dataset.values, num_test_days)

        train_data = trainData[:, self.start_col: self.start_col + self.number_of_features]
        test_data = testData[:, self.start_col: self.start_col + self.number_of_features]

        self.train_dates = dateTime[: -(num_test_days * 24):]
        self.test_dates = dateTime[-(num_test_days * 24):]

        # print("TrainData shape: ", train_data .shape) # days x hour x features (12264, 6)
        # print("TestData shape: ", test_data.shape) # days x hour x features (5280, 6) splitWeatherDataset_v2

        # make sure the weather data is the same length as the dataset
        weatherDataset = weatherDataset[: dataset_time_step]
        weahter_train_data, weahter_test_data = splitWeatherDataset_v2(weatherDataset.values, num_test_days)

        weahter_train_data = weahter_train_data[:, :self.num_weather_features]
        weahter_test_data = weahter_test_data[:, :self.num_weather_features]

        # print("WeatherTrainData shape: ", weahter_train_data.shape) # (days x hour) x features (12264, 7)
        # print("WeatherTestData shape: ", weahter_test_data.shape) # (days x hour) x features  (5280, 7)

        train_data = fillMissingData(train_data)
        test_data = fillMissingData(test_data)

        weahter_train_data = fillMissingData(weahter_train_data)
        weahter_test_data = fillMissingData(weahter_test_data)

        # print("***** Dataset split done *****")

        featureList = dataset.columns.values
        featureList = featureList[self.start_col: self.start_col + self.number_of_features].tolist()
        featureList.extend(weatherDataset.columns.values[:self.number_of_features])
        print(f"\nFeatures: {featureList}\n")

        # print("Scaling data...")

        train_data, test_data, self.ft_min, self.ft_max = scaleDataset(train_data, test_data)
        # print(train_data.shape, test_data.shape)  # (12264, 6) (5280, 6)

        weahter_train_data, weahter_test_data, self.w_ft_min, self.w_ft_max = scaleDataset(weahter_train_data, weahter_test_data)
        # print(weahter_train_data.shape, weahter_test_data.shape)  # (12264, 7) (5280, 7)
        # print("***** Data scaling done *****\n")

        if self.data_split == "train":
            self.data = train_data
            self.weather_data = weahter_train_data

        elif self.data_split == "test":
            self.data = test_data
            self.weather_data = weahter_test_data

        self.feature_x, self.weather_feature_x, self.output_y = manipulateTrainingDataShape_v2(self.data, self.carbon_intensity_index, 
                                                        self.seq_len, self.forecast_horizon, 
                                                        sliding_windows=self.data_stride_len, 
                                                        weatherData=self.weather_data)


    def __getitem__(self, index):

        feature_x = self.feature_x[index]
        weather_feature_x = self.weather_feature_x[index]

        forecast = self.output_y[index]

        return feature_x, weather_feature_x, forecast


    def __len__(self):

        return self.feature_x.shape[0]

    def unscale_bak(self, data):

        max_v = self.ft_max[self.carbon_intensity_index]
        min_v = self.ft_min[self.carbon_intensity_index]

        cdiff = max_v - min_v
        unscaledData = np.zeros_like(data)

        for i in range(data.shape[0]):
            unscaledData[i] = round(max(data[i] * cdiff +  min_v, 0), 5)

        return unscaledData
    
    def unscale(self, data):
        original_shape = data.shape

        data = data.reshape(-1)

        max_v = self.ft_max[self.carbon_intensity_index]
        min_v = self.ft_min[self.carbon_intensity_index]

        cdiff = max_v - min_v

        # Vectorized unscale and clipping to zero minimum, then rounding
        unscaledData = np.round(np.clip(data * cdiff + min_v, 0, None), 5)

        unscaledData = unscaledData.reshape(original_shape)

        return unscaledData


class CarbonDataset_BASE_v2:
    def __init__(self,
                 data_base_dir: str = "/kang/carbon/CarbonCast_main/data",
                 region: str = "CISO",
                 carbon_type: str = "lifecycle",
                 input_seq_len: int = 512,
                 forecast_horizon: int = 192,
                 data_split: str = "train",
                 data_stride_len: int = 1, 
                 extra_features_time: bool = False, 
                 extra_features_weather: bool = False, 
                 only_carbon_forecast: bool = False):

        start_col = 1   # index in inFileName
        number_of_features = 6  # carbon_intensity, plus 5 time-related features: (hour_sin, hour_cos, month_sin, month_cos, weekend)
        num_weather_features = 7  # in carboncast the valuse is 12, since it contains source prediction electricty from first tier model
        carbon_intensity_index = 0  # carbon_intensity index in dataset

        self.extra_features_time = extra_features_time
        self.extra_features_weather = extra_features_weather

        self.only_carbon_forecast = only_carbon_forecast

        self.start_col = start_col
        self.number_of_features = number_of_features
        self.num_weather_features = num_weather_features
        self.carbon_intensity_index = carbon_intensity_index
        
        data_base_dir_region = os.path.join(data_base_dir, region)

        # '/kang/carbon/CarbonCast_main/data/CISO/CISO_direct_emissions.csv'
        in_file_name = os.path.join(data_base_dir_region, f'{region}_{carbon_type}_emissions.csv')
        self.in_file_name = in_file_name

        # '/mnt/ssd2/kangyang/carbon/TSFM_Building_carbon/data/weather/CISO_weather_2020_2021.csv'
        # weather_file_name = f"/mnt/ssd2/kangyang/carbon/TSFM_Building_carbon/data/weather/{region}_weather_2020_2021.csv"
        weather_file_name = f"/kang/carbon/tsfm_carbon/data/weather/{region}_weather_2020_2021.csv"

        self.weather_file_name = weather_file_name

        # self.predifined_input_length = 512

        self.seq_len = input_seq_len
        self.forecast_horizon = forecast_horizon
        # self.full_file_path_and_name = data_path
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        # self.random_seed = random_seed

        training_ratio=0.7
        # Read data
        self._read_data(training_ratio=training_ratio)


    def _get_borders(self):
        total_len = self.length_timeseries_original
        train_pct = 0.70  # for days

        num_days = total_len // 24
        num_train_days = int(num_days * train_pct)
        num_test_days = num_days - num_train_days

        # train_secdond_ratio = 0.01
        # num_train_days = int(num_train_days * train_secdond_ratio)
        # if num_train_days < 1:
        #     num_train_days = 1

        n_train = int(num_train_days * 24)
        n_val = 0
        n_test = int(num_test_days * 24)

        # n_train = 12 * 30 * 24
        # n_val = 4 * 30 * 24
        # n_test = 4 * 30 * 24

        train_end = n_train
        val_end = n_train + n_val

        test_start = val_end - self.seq_len   # # allow window to start earlier
        test_end = test_start + n_test + self.seq_len

        train = slice(0, train_end)
        test = slice(test_start, test_end)

        return train, test
    

    def _read_data(self, training_ratio=0.7):
        self.scaler = StandardScaler()
        self.scaler_weahter = StandardScaler()

        dataset, weatherDataset, self.date_time = initialize(self.in_file_name, self.weather_file_name, self.start_col)

        # this is for save the dataset, not used in this function, but in the save_dataset function
        self.dataset_save = dataset.values[:, self.start_col: self.start_col + self.number_of_features]
        self.weather_dataset_save = weatherDataset

        dataset_time_step, feature_len = dataset.shape
        self.length_timeseries_original = dataset_time_step

        num_days = dataset_time_step // 24
        # training_ratio = 0.7
        # testing_ratio = 1 - training_ratio

        num_train_days = int(num_days * training_ratio)
        num_test_days = num_days - num_train_days  # 220

        train_secdond_ratio = 1
        num_train_days = int(num_train_days * train_secdond_ratio)
        if num_train_days < 1:
            num_train_days = 1

        print(f"ratio: {train_secdond_ratio * 100}, traininig steps: {num_train_days * 24}")
        # split into train and test
        # print("Spliting dataset into train/test...")
        # trainData: (12264, 6), testData: (5280, 6)
        trainData, testData = splitDataset(dataset.values, num_train_days, num_test_days, self.seq_len)

        train_data = trainData[:, self.start_col: self.start_col + self.number_of_features]
        test_data = testData[:, self.start_col: self.start_col + self.number_of_features]

        train_dates, test_dates = splitDataset(self.date_time, num_train_days, num_test_days, self.seq_len)
        # self.train_dates = self.date_time[: -(num_test_days * 24):]
        # self.test_dates = self.date_time[-(num_test_days * 24):]

        # print("TrainData shape: ", train_data .shape) # days x hour x features (12264, 6)
        # print("TestData shape: ", test_data.shape) # days x hour x features (5280, 6) splitWeatherDataset_v2

        # make sure the weather data is the same length as the dataset
        weatherDataset = weatherDataset[: dataset_time_step]
        # weahter_train_data, weahter_test_data = splitWeatherDataset_v2(weatherDataset.values, num_test_days)
        weahter_train_data, weahter_test_data = splitDataset(weatherDataset.values, num_train_days, num_test_days, self.seq_len)

        weahter_train_data = weahter_train_data[:, :self.num_weather_features]
        weahter_test_data = weahter_test_data[:, :self.num_weather_features]

        # print("WeatherTrainData shape: ", weahter_train_data.shape) # (days x hour) x features (12264, 7)
        # print("WeatherTestData shape: ", weahter_test_data.shape) # (days x hour) x features  (5280, 7)

        train_data = fillMissingData(train_data)
        test_data = fillMissingData(test_data)

        weahter_train_data = fillMissingData(weahter_train_data)
        weahter_test_data = fillMissingData(weahter_test_data)

        featureList = dataset.columns.values
        featureList = featureList[self.start_col: self.start_col + self.number_of_features].tolist()
        featureList.extend(weatherDataset.columns.values[:self.num_weather_features])
        # print(f"\nFeatures: {featureList}\n")

        self.feature_list = featureList

        train_data_scaled = self.scaler.fit_transform(train_data)
        test_data_scaled = self.scaler.transform(test_data)

        weahter_train_data_scaled = self.scaler_weahter.fit_transform(weahter_train_data)
        weahter_test_data_scaled = self.scaler_weahter.transform(weahter_test_data)

        # train_data, test_data, self.ft_min, self.ft_max = scaleDataset(train_data, test_data)
        # print(train_data.shape, test_data.shape)  # (12264, 6) (5280, 6)

        # weahter_train_data, weahter_test_data, self.w_ft_min, self.w_ft_max = scaleDataset(weahter_train_data, weahter_test_data)
        # print(weahter_train_data.shape, weahter_test_data.shape)  # (12264, 7) (5280, 7)
        # print("***** Data scaling done *****\n")

        if self.data_split == "train":
            self.data = train_data_scaled
            self.weather_data = weahter_train_data_scaled
            self.date_time_split = train_dates

        elif self.data_split == "test":
            self.data = test_data_scaled
            self.weather_data = weahter_test_data_scaled
            self.date_time_split = test_dates

        self.length_timeseries = self.data.shape[0]

        # self.feature_x, self.weather_feature_x, self.output_y = manipulateTrainingDataShape_v2(self.data, self.carbon_intensity_index, 
        #                                                 self.seq_len, self.forecast_horizon, 
        #                                                 sliding_windows=self.data_stride_len, 
        #                                                 weatherData=self.weather_data)


    def __getitem_bakup__(self, index):

        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len

        pred_end = seq_end + self.forecast_horizon

        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_end = seq_end - self.forecast_horizon
            seq_start = seq_end - self.seq_len    

        timeseries = self.data[seq_start:seq_end, :]
        forecast = self.data[seq_end:pred_end, self.carbon_intensity_index]

        weather_feature_x = self.weather_data[seq_end:pred_end, :]

        return timeseries, weather_feature_x, forecast
    
    def __getitem__(self, index):
        pass

    def __len__(self):

        # return self.feature_x.shape[0]
        return (self.length_timeseries - self.seq_len - self.forecast_horizon) // self.data_stride_len + 1


    def unscale_bak(self, data):

        max_v = self.ft_max[self.carbon_intensity_index]
        min_v = self.ft_min[self.carbon_intensity_index]

        cdiff = max_v - min_v
        unscaledData = np.zeros_like(data)

        for i in range(data.shape[0]):
            unscaledData[i] = round(max(data[i] * cdiff +  min_v, 0), 5)

        return unscaledData
    
    def unscale(self, data):
        original_shape = data.shape

        data = data.reshape(-1)

        max_v = self.ft_max[self.carbon_intensity_index]
        min_v = self.ft_min[self.carbon_intensity_index]

        cdiff = max_v - min_v

        # Vectorized unscale and clipping to zero minimum, then rounding
        unscaledData = np.round(np.clip(data * cdiff + min_v, 0, None), 5)

        unscaledData = unscaledData.reshape(original_shape)

        return unscaledData


class CarbonDataset_BASE_v3:
    def __init__(self,
                 data_base_dir: str = "/kang/carbon/CarbonCast_main/data",
                 region: str = "CISO",
                 carbon_type: str = "lifecycle",
                 input_seq_len: int = 512,
                 forecast_horizon: int = 192,
                 data_split: str = "train",
                 data_stride_len: int = 1, 
                 extra_features_time: bool = False, 
                 extra_features_weather: bool = False, 
                 only_carbon_forecast: bool = False):

        self.carbon_intensity_index = 5

        self.number_of_features = 1
        
        region_kv = {"Argentina": "AR", 
                     "Chile": "CL-SEN", 
                     "India-North": "IN-NO", 
                     "India-South": "IN-SO",
                     "New-Zealand": "NZ",
                     "Nigeria": "NG", 
                     "Taiwan": "TW",
                     "Uruguay": "UY"}

        year_list = [2021, 2022, 2023, 2024]
        data_base_dir_region = os.path.join(data_base_dir, region)

        in_file_name_list = []
        region_v = region_kv.get(region)
        for year_i in year_list:
            in_file_name = os.path.join(data_base_dir_region, f'{region_v}_{year_i}_hourly.csv')

            in_file_name_list.append(in_file_name)

        self.in_file_name_list = in_file_name_list

        self.seq_len = input_seq_len
        self.forecast_horizon = forecast_horizon
        self.data_split = data_split
        self.data_stride_len = data_stride_len

        training_ratio=0.7

        self._read_data(training_ratio=training_ratio)


    def _get_borders(self):
        total_len = self.length_timeseries_original
        train_pct = 0.70  # for days

        num_days = total_len // 24
        num_train_days = int(num_days * train_pct)
        num_test_days = num_days - num_train_days

        # train_secdond_ratio = 0.01
        # num_train_days = int(num_train_days * train_secdond_ratio)
        # if num_train_days < 1:
        #     num_train_days = 1

        n_train = int(num_train_days * 24)
        n_val = 0
        n_test = int(num_test_days * 24)

        # n_train = 12 * 30 * 24
        # n_val = 4 * 30 * 24
        # n_test = 4 * 30 * 24

        train_end = n_train
        val_end = n_train + n_val

        test_start = val_end - self.seq_len   # # allow window to start earlier
        test_end = test_start + n_test + self.seq_len

        train = slice(0, train_end)
        test = slice(test_start, test_end)

        return train, test
    

    def _initialize(self):

        print(self.in_file_name_list)

        # Read and combine all CSV files into a single DataFrame
        dataset = pd.concat([
            pd.read_csv(inFileName, header=0, parse_dates=['Datetime (UTC)'], index_col=['Datetime (UTC)'])
            for inFileName in self.in_file_name_list
        ])

        dateTime = dataset.index.values
        carbon_intensity_data = dataset[['Carbon intensity gCO₂eq/kWh (Life cycle)']].astype(np.float64)

        return carbon_intensity_data, dateTime


    def _read_data(self, training_ratio=0.7):
        self.scaler = StandardScaler()
        self.scaler_weahter = StandardScaler()

        dataset, self.date_time = self._initialize()

        dataset_time_step, feature_len = dataset.shape
        self.length_timeseries_original = dataset_time_step

        num_days = dataset_time_step // 24
        # training_ratio = 0.7
        # testing_ratio = 1 - training_ratio

        num_train_days = int(num_days * training_ratio)
        num_test_days = num_days - num_train_days  # 220

        print(f"num_train_days: {num_train_days}, num_test_days: {num_test_days}")

        train_secdond_ratio = 1
        num_train_days = int(num_train_days * train_secdond_ratio)
        if num_train_days < 1:
            num_train_days = 1

        # print(f"ratio: {train_secdond_ratio * 100}, traininig steps: {num_train_days * 24}")

        train_data, test_data = splitDataset(dataset.values, num_train_days, num_test_days, self.seq_len)

        self.dataset_save = dataset.values
        # self.weather_dataset_save = weatherDataset

        # train_data = trainData[:, self.start_col: self.start_col + self.number_of_features]
        # test_data = testData[:, self.start_col: self.start_col + self.number_of_features]

        train_dates, test_dates = splitDataset(self.date_time, num_train_days, num_test_days, self.seq_len)

        train_data = fillMissingData(train_data)
        test_data = fillMissingData(test_data)

        train_data_scaled = self.scaler.fit_transform(train_data)
        test_data_scaled = self.scaler.transform(test_data)

        if self.data_split == "train":
            self.data = train_data_scaled
            self.date_time_split = train_dates

        elif self.data_split == "test":
            self.data = test_data_scaled
            self.date_time_split = test_dates

        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):

        # index = 2211
        steps, data_feature_len = self.data.shape  # steps: 5304, data_feature_len: 6

        seq_start = self.data_stride_len * index  # 2211
        seq_end = seq_start + self.seq_len        # 2235

        # featture_all = np.concatenate((self.data, self.weather_data), axis=1)  # (X, 6 + 7 = 13)

        pred_end = seq_end + self.forecast_horizon   # 2331

        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_end = seq_end - self.forecast_horizon
            seq_start = seq_end - self.seq_len

        timeseries = self.data[seq_start: seq_end, :]  # (24, 13)
        
        forecast = self.data[seq_end: pred_end, :]  # (96, 13)

        feature_x = timeseries  

        predifined_input_length_moment = 512
        if self.seq_len < predifined_input_length_moment:
            pad_len = predifined_input_length_moment - self.seq_len  # self.seq_len = 24, pad_len = 488
        else:
            pad_len = 0

        padding = np.zeros((pad_len, feature_x.shape[1]))  # (488, 13)
        feature_x = np.vstack((feature_x, padding))  # (512, 13)              # shape becomes [512, features]

        input_mask = np.concatenate((np.ones(self.seq_len), np.zeros(pad_len)))  # (512,)

        feature_x = feature_x.T  # (13, 512)
        forecast = forecast.T    # (13, 96)

        # forecast = np.expand_dims(forecast, axis=0)  # need to be (1, 96)

        return feature_x, forecast, input_mask


    def __len__(self):

        # return self.feature_x.shape[0]
        return (self.length_timeseries - self.seq_len - self.forecast_horizon) // self.data_stride_len + 1


    def unscale_bak(self, data):

        max_v = self.ft_max[self.carbon_intensity_index]
        min_v = self.ft_min[self.carbon_intensity_index]

        cdiff = max_v - min_v
        unscaledData = np.zeros_like(data)

        for i in range(data.shape[0]):
            unscaledData[i] = round(max(data[i] * cdiff +  min_v, 0), 5)

        return unscaledData
    
    def unscale(self, data):
        original_shape = data.shape

        data = data.reshape(-1)

        max_v = self.ft_max[self.carbon_intensity_index]
        min_v = self.ft_min[self.carbon_intensity_index]

        cdiff = max_v - min_v

        # Vectorized unscale and clipping to zero minimum, then rounding
        unscaledData = np.round(np.clip(data * cdiff + min_v, 0, None), 5)

        unscaledData = unscaledData.reshape(original_shape)

        return unscaledData


    def save_dataset(self, save_path):
        """
        Save the dataset to a file.
        """

        self.weather_dataset_save = self.dataset_save
        # Combine main and weather features
        feature_all = np.concatenate((self.dataset_save, self.weather_dataset_save), axis=1)  # (X, 13)

        # Move the first column to the last position
        # feature_all = np.concatenate((feature_all[:, 1:], feature_all[:, [0]]), axis=1)


        # Ensure date_time is 2D column and format as string
        date_time = np.array([pd.to_datetime(dt).strftime('%Y-%m-%d %H:%M:%S') for dt in self.date_time]).reshape(-1, 1)

        # Combine date column with feature data
        feature_all = np.concatenate((date_time, feature_all), axis=1)  # (X, 14)

        # Prepare column headers
        feature_list = ["carbon_intensity", "carbon_intensity"]
        # Move the first item to the end of the feature list
        # feature_list = feature_list[1:] + [feature_list[0]]

        feature_list.insert(0, "date")

        # Sanity checks
        assert feature_all.shape[0] == len(self.date_time), "Date time length does not match data rows."
        assert feature_all.shape[1] == len(feature_list), "Feature list length does not match number of columns."

        # Build DataFrame and save
        df = pd.DataFrame(feature_all, columns=feature_list)
        df.to_csv(save_path, index=False)



class CarbonDataset_MOMENT(CarbonDataset_BASE_v2):
    def __getitem__(self, index):

        # index = 2211
        # steps, data_feature_len = self.data.shape
        # weather_steps, weather_feature_len = self.weather_data.shape

        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len

        pred_end = seq_end + self.forecast_horizon

        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_end = seq_end - self.forecast_horizon
            seq_start = seq_end - self.seq_len

        timeseries = self.data[seq_start: seq_end, :]  # (24, 6)
        
        weather_feature_x = self.weather_data[seq_end:pred_end, :]  # (96, 7)
    
        if self.extra_features_time and self.extra_features_weather:
            feature_x = np.concatenate((timeseries, weather_feature_x), axis=1)  # (512, 6 + 7 = 13)

            # forcast_index = data_feature_len

        elif self.extra_features_weather:
            feature_x = np.concatenate((timeseries[:, 0:1], weather_feature_x), axis=1)  # (512, 1 + 7 = 8)

            # forcast_index = self.carbon_intensity_index + 1

        elif self.extra_features_time:
            feature_x = timeseries  # (512, 6)

            # forcast_index = data_feature_len
            # forecast = self.data[seq_end: pred_end, :]

        else:
            feature_x = timeseries[:, 0:1]  # (512, 1)

            # forcast_index = self.carbon_intensity_index + 1 # 1

        # if self.only_carbon_forecast:
        forcast_index = self.carbon_intensity_index + 1

        forecast = self.data[seq_end: pred_end, 0: forcast_index]  # (96, 1)

        # MOMENT requires the input timeseries to be of length 512
        predifined_input_length_moment = 512
        if self.seq_len < predifined_input_length_moment:
            pad_len = predifined_input_length_moment - self.seq_len  # self.seq_len = 24, pad_len = 488
        else:
            pad_len = 0

        padding = np.zeros((pad_len, feature_x.shape[1]))  # (488, 6)
        feature_x = np.vstack((feature_x, padding))  # (512, 6)              # shape becomes [512, features]

        input_mask = np.concatenate((np.ones(self.seq_len), np.zeros(pad_len)))  # (512,)

        feature_x = feature_x.T  # (13, 512)
        forecast = forecast.T    # (1, 96)

        # forecast = np.expand_dims(forecast, axis=0)  # need to be (1, 96)

        return feature_x, forecast, input_mask


class CarbonDataset_MOMENT_v2(CarbonDataset_BASE_v2):
    def __getitem__(self, index):

        # index = 2211
        steps, data_feature_len = self.data.shape  # steps: 5304, data_feature_len: 6
        weather_steps, weather_feature_len = self.weather_data.shape  # weather_steps: 5304, weather_feature_len: 7

        seq_start = self.data_stride_len * index  # 2211
        seq_end = seq_start + self.seq_len        # 2235

        featture_all = np.concatenate((self.data, self.weather_data), axis=1)  # (X, 6 + 7 = 13)

        pred_end = seq_end + self.forecast_horizon   # 2331

        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_end = seq_end - self.forecast_horizon
            seq_start = seq_end - self.seq_len

        timeseries = featture_all[seq_start: seq_end, :]  # (24, 13)
        
        forecast = featture_all[seq_end: pred_end, :]  # (96, 13)

        # weather_feature_x = self.weather_data[seq_start:seq_end, :]  # (24, 7)
    
        if self.extra_features_time and self.extra_features_weather:
            feature_x = timeseries  # (X, 6 + 7 = 13)

            if self.only_carbon_forecast:
                forecast = forecast[:, 0: 1]  # (96, 1)

        elif self.extra_features_weather:
            feature_x = np.concatenate((timeseries[:, 0:1], timeseries[:, data_feature_len:]), axis=1)  # (512, 1 + 7 = 8)

            if self.only_carbon_forecast:
                forecast = forecast[:, 0: 1]  # (96, 1)

            else:
                carbon = forecast[:, 0:1]  # (96, 1)
                weather = forecast[:, data_feature_len:]
                forecast = np.concatenate((carbon, weather), axis=1)  # (96, 1 + 7 = 8)
            # forcast_index = self.carbon_intensity_index + 1

        elif self.extra_features_time:
            feature_x = timeseries  # (512, 6)

            if self.only_carbon_forecast:
                forecast = forecast[:, 0: 1]  # (96, 1)
            else:
                forecast = forecast[:, 0: data_feature_len]
            # forcast_index = data_feature_len
            # forecast = self.data[seq_end: pred_end, :]

        else:
            feature_x = timeseries[:, 0: 1]  # (512, 1)

            forecast = forecast[:, 0: 1]  # (96, 1)

            # forcast_index = self.carbon_intensity_index + 1 # 1

        # if self.only_carbon_forecast:
        # forcast_index = self.carbon_intensity_index + 1

        # forecast = self.data[seq_end: pred_end, 0: forcast_index]  # (96, 1)

        # MOMENT requires the input timeseries to be of length 512
        predifined_input_length_moment = 512
        if self.seq_len < predifined_input_length_moment:
            pad_len = predifined_input_length_moment - self.seq_len  # self.seq_len = 24, pad_len = 488
        else:
            pad_len = 0

        padding = np.zeros((pad_len, feature_x.shape[1]))  # (488, 13)
        feature_x = np.vstack((feature_x, padding))  # (512, 13)              # shape becomes [512, features]

        input_mask = np.concatenate((np.ones(self.seq_len), np.zeros(pad_len)))  # (512,)

        feature_x = feature_x.T  # (13, 512)
        forecast = forecast.T    # (13, 96)

        # forecast = np.expand_dims(forecast, axis=0)  # need to be (1, 96)

        return feature_x, forecast, input_mask


    def save_dataset(self, save_path):
        """
        Save the dataset to a file.
        """

        # Combine main and weather features
        feature_all = np.concatenate((self.dataset_save, self.weather_dataset_save), axis=1)  # (X, 13)

        # Move the first column to the last position
        feature_all = np.concatenate((feature_all[:, 1:], feature_all[:, [0]]), axis=1)


        # Ensure date_time is 2D column and format as string
        date_time = np.array([pd.to_datetime(dt).strftime('%Y-%m-%d %H:%M:%S') for dt in self.date_time]).reshape(-1, 1)

        # Combine date column with feature data
        feature_all = np.concatenate((date_time, feature_all), axis=1)  # (X, 14)

        # Prepare column headers
        feature_list = self.feature_list.copy()
        # Move the first item to the end of the feature list
        feature_list = feature_list[1:] + [feature_list[0]]

        feature_list.insert(0, "date")

        # Sanity checks
        assert feature_all.shape[0] == len(self.date_time), "Date time length does not match data rows."
        assert feature_all.shape[1] == len(feature_list), "Feature list length does not match number of columns."

        # Build DataFrame and save
        df = pd.DataFrame(feature_all, columns=feature_list)
        df.to_csv(save_path, index=False)


    def save_dataset_single(self, save_path):
        """
        Save the dataset to a file.
        """

        # Combine main and weather features
        feature_all = self.dataset_save[:, 0:1]  # (X, 1)

        # Move the first column to the last position
        # feature_all = np.concatenate((feature_all[:, 1:], feature_all[:, [0]]), axis=1)


        # Ensure date_time is 2D column and format as string
        date_time = np.array([pd.to_datetime(dt).strftime('%Y-%m-%d %H:%M:%S') for dt in self.date_time]).reshape(-1, 1)

        # Combine date column with feature data
        feature_all = np.concatenate((date_time, feature_all), axis=1)  # (X, 2)

        # Prepare column headers
        feature_list = self.feature_list.copy()[0: 1]
        # Move the first item to the end of the feature list
        # feature_list = feature_list[1:] + [feature_list[0]]

        feature_list.insert(0, "date")

        # Sanity checks
        assert feature_all.shape[0] == len(self.date_time), "Date time length does not match data rows."
        assert feature_all.shape[1] == len(feature_list), "Feature list length does not match number of columns."

        # Build DataFrame and save
        df = pd.DataFrame(feature_all, columns=feature_list)
        df.to_csv(save_path, index=False)


class CarbonDataset_CHRONOS(CarbonDataset_BASE_v2):
    
    def __getitem__(self, index):

        # index = 2211
        steps, data_feature_len = self.data.shape  # steps: 5304, data_feature_len: 6
        weather_steps, weather_feature_len = self.weather_data.shape  # weather_steps: 5304, weather_feature_len: 7

        seq_start = self.data_stride_len * index  # 2211
        seq_end = seq_start + self.seq_len        # 2235

        featture_all = np.concatenate((self.data, self.weather_data), axis=1)  # (X, 6 + 7 = 13)

        pred_end = seq_end + self.forecast_horizon   # 2331

        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_end = seq_end - self.forecast_horizon
            seq_start = seq_end - self.seq_len

        timeseries = featture_all[seq_start: seq_end, :]  # (24, 13)
        
        forecast = featture_all[seq_end: pred_end, :]  # (96, 13)

        # weather_feature_x = self.weather_data[seq_start:seq_end, :]  # (24, 7)
    
        if self.extra_features_time and self.extra_features_weather:
            feature_x = timeseries  # (X, 6 + 7 = 13)

            if self.only_carbon_forecast:
                forecast = forecast[:, 0: 1]  # (96, 1)

        elif self.extra_features_weather:
            feature_x = np.concatenate((timeseries[:, 0:1], timeseries[:, data_feature_len:]), axis=1)  # (512, 1 + 7 = 8)

            if self.only_carbon_forecast:
                forecast = forecast[:, 0: 1]  # (96, 1)

            else:
                carbon = forecast[:, 0:1]  # (96, 1)
                weather = forecast[:, data_feature_len:]
                forecast = np.concatenate((carbon, weather), axis=1)  # (96, 1 + 7 = 8)
            # forcast_index = self.carbon_intensity_index + 1

        elif self.extra_features_time:
            feature_x = timeseries  # (512, 6)

            if self.only_carbon_forecast:
                forecast = forecast[:, 0: 1]  # (96, 1)
            else:
                forecast = forecast[:, 0: data_feature_len]
            # forcast_index = data_feature_len
            # forecast = self.data[seq_end: pred_end, :]

        else:
            feature_x = timeseries[:, 0: 1]  # (512, 1)

            forecast = forecast[:, 0: 1]  # (96, 1)

        predifined_input_length_moment = 512
        if self.seq_len < predifined_input_length_moment:
            pad_len = predifined_input_length_moment - self.seq_len  # self.seq_len = 24, pad_len = 488
        else:
            pad_len = 0

        # padding = np.zeros((pad_len, feature_x.shape[1]))  # (488, 13)
        # feature_x = np.vstack((feature_x, padding))  # (512, 13)              # shape becomes [512, features]

        input_mask = np.concatenate((np.ones(self.seq_len), np.zeros(pad_len)))  # (512,)

        feature_x = feature_x.T  # (13, 512)
        forecast = forecast.T    # (13, 96)

        # forecast = np.expand_dims(forecast, axis=0)  # need to be (1, 96)

        return feature_x, forecast, input_mask


class CarbonDataset_TIMEGPT(CarbonDataset_BASE_v2):
    
    def __getitem__(self, index):

        # date_time = np.array([pd.to_datetime(dt).strftime('%Y-%m-%d %H:%M:%S') for dt in self.date_time_split]).reshape(-1, 1)
        date_time = [pd.to_datetime(dt).strftime('%Y-%m-%d %H:%M:%S') for dt in self.date_time_split]


        featture_all = self.data[:, 0:1]

        seq_start = self.data_stride_len * index  
        seq_end = seq_start + self.seq_len        

        pred_end = seq_end + self.forecast_horizon  

        if pred_end > self.length_timeseries:
            pred_end = self.length_timeseries
            seq_end = seq_end - self.forecast_horizon
            seq_start = seq_end - self.seq_len

        timeseries = featture_all[seq_start: seq_end, :]

        # date_time_s = date_time[seq_start: seq_end, :]
        date_time_s = date_time[seq_start: seq_end]  # just a list

        forecast = featture_all[seq_end: pred_end, :]  

        forecast = forecast.T    # (13, 96)
        timeseries = timeseries.T  # (1, 24)
        # date_time_s = date_time_s.T

        return {
            "timeseries": timeseries,
            "forecast": forecast,
            "date_time": date_time_s
        }



if __name__ == "__main__":
    
    # region_list = ["CISO", "PJM", "ERCO", "ISNE", "SE", "DE"]
    region_list = ["BPAT", "FPL", "NYISO", "PL", "ES", "NL", "AUS_QLD"]

    region_list_1 = ["CISO", "PJM", "ERCO", "ISNE", "BPAT", "FPL", "NYISO", "SE", "DE", "PL", "ES", "NL", "AUS_QLD"]

    region_list_2 = ['Argentina', 'Chile', 'India-North', 'India-South', 'New-Zealand', 'Nigeria', 'Taiwan', 'Uruguay']


    data_base_dir_t = "/kang/carbon/CarbonCast_main/data"
    carbon_type_t = "lifecycle"
    input_seq_len_t = 24
    forecast_horizon_t = 96
    data_split_t = "test"
    data_stride_len_t = 1
    extra_f_time = True
    extra_f_weather = True
    # if extra_f_weather:
    #     input_seq_len_t = forecast_horizon_t  # because weather if forecasted, its length is the same as forecast horizon

    # make it always False, cut the extra features when training or testing
    only_carbon_forecast = False

    for region_t in region_list_2:
        # region_t = "Argentina"


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
            

        base_dir = "/kang/carbon/Time-Series-Library/dataset"
        save_dir = os.path.join(base_dir, "carbon")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{region_t}_{carbon_type_t}_dataset.csv")
        test_dataset.save_dataset(save_path)

        # print(f"Dataset saved to {save_path}")

        # save_path = os.path.join(save_dir, f"{region_t}_{carbon_type_t}_dataset_single.csv")
        # test_dataset.save_dataset_single(save_path)