
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import os
from datasets import load_dataset
from datasets import load_dataset
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def get_carbon_data(duration, pred_hrz, sampling_rate, house_id=1, data_df=None):
    # from config import elec_uci_indices, elec_uci_season
    # id_num = house_id
    # assert house_id >= 1 and house_id <= 370
    # house_id = str(house_id)
    # house_id = 'MT_'+'0'*(3-len(house_id)) + house_id
    # Read the data, considering the specific delimiter and the first line as header
    # df = pd.read_csv(file_path, delimiter=';', header=0)
    house_id = "carbon_intensity"
    assert data_df is not None
    df = data_df.copy()
    df.rename(columns={df.columns[1]: 'TS'}, inplace=True) # rename the first column of a DataFrame (df) to 'TS'

    df = df[['TS', house_id]] # selecting just two columns from the DataFrame df

    # Set MT_001 to float
    # df[house_id] = df[house_id].str.replace(',', '.').astype(float) # replaces commas with dots in all string values in that column. like "1,23" to "1.23"
    df[house_id] = df[house_id].astype(float)     # replaces commas with dots in all string values in that column. like "1,23" to "1.23"

    df['TS'] = pd.to_datetime(df['TS']) # Converts the values to Pandas datetime objects

    # batch_idx = batch_id
    # print(sample_by_season(df))
    # pdb.set_trace()
    # csv_path = './data/uci_indices.csv'
    # df_indices = pd.read_csv(csv_path)

    # df = df.iloc[df_indices[str(id_num)][batch_id]:-1,:] # start_index = df_indices[str(id_num)][batch_id]  # e.g., 48294, df = df.iloc[start_index:-1, :]
    # df.iloc[start_index:-1, :]: Slice the DataFrame from that row to the second-to-last row (because -1 excludes the last row in .iloc), :: Keep all columns

    df.set_index('TS', inplace=True) # Now 'TS' is the index, not a column.

    # resampled_df = df.resample(f'{sampling_rate}s').asfreq().fillna(0).reset_index()
    resampled_df = df.resample(f'{sampling_rate}s').mean().interpolate(method='time').fillna(0).reset_index()
    # df.resample(f'{sampling_rate}s'): Groups your time series data into bins every sampling_rate seconds,  if sampling_rate = 5, this creates 5-second intervals
    # .mean(): For each interval, it computes the mean of all values (averages the data in each time window)
    # .interpolate(method='time'): Fills in any missing values between time steps using linear interpolation based on timestamps
    # .fillna(0): If any NaNs still remain (e.g., before the first/after the last time value), fill them with 0
    # .reset_index(): Moves the time index (TS) back into a regular column, so resampled_df looks like a typical DataFrame

    # search for the first non-zero row
    first_non_zero_index = resampled_df[resampled_df[house_id] != 0].index[0]
    # resampled_df[house_id] != 0, Creates a boolean mask — True where the value is not zero. resampled_df[...], Filters the DataFrame to only rows where that column is non-zero.
    # .index[0]: Gets the index of the first non-zero row.

    # Calculate the length of data needed

    start_points = first_non_zero_index

    return resampled_df, start_points


def get_carbon_batch_data(resampled_df, start_points, len_data, len_gt, batch_id=0):
    house_id = "carbon_intensity"

    # Shift the starting point by batch offset
    offset = batch_id * (len_data + len_gt)
    batch_start = start_points + offset

    # Slice the input and prediction windows
    batch_data = resampled_df[[house_id, 'TS']].values[batch_start : batch_start + len_data]
    batch_test_data = resampled_df[[house_id, 'TS']].values[batch_start + len_data : batch_start + len_data + len_gt]

    # batch_data = resampled_df[[house_id, 'TS']].values[start_points: start_points + len_data]
    # batch_test_data = resampled_df[[house_id, 'TS']].values[start_points + len_data: start_points + len_data + len_gt]

    return batch_data, batch_test_data


def generate_datetime_list(start_datetime, increase, num_steps, offset=0):
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta 
    # Ensure start_datetime is a pandas Timestamp and has time components
    if not isinstance(start_datetime, pd.Timestamp):
        raise ValueError("start_datetime must be a pandas Timestamp.")
    
    # Set time to 00:00:00 if start_datetime does not include hours, minutes, and seconds
    if start_datetime.hour == 0 and start_datetime.minute == 0 and start_datetime.second == 0:
        start_datetime = start_datetime.replace(hour=0, minute=0, second=0)
    
    # Extract the number and unit from 'increase'
    if increase[:-1].isdigit():
        n = int(increase[:-1])
        unit = increase[-1]
    else:
        n = 1  # Default increment if no number is provided
        unit = increase

    # Determine the increment based on the unit
    if unit == 'H':
        increment = timedelta(hours=n)
    elif unit == 'T':
        increment = timedelta(minutes=n)
    elif unit == 'D':
        increment = timedelta(days=n)
    elif unit == 'M':
        increment = relativedelta(months=n)
    elif unit == 'A-DEC':
        increment = relativedelta(years=n)
    elif unit == 'W-SUN':
        increment = relativedelta(weeks=n)
    else:
        raise ValueError("Invalid increase value. Must be in ['H', 'T', 'D', 'M', 'A-DEC', 'W-SUN'] with optional 'n' prefix.")

    # Generate the list of datetime values
    datetime_list = []
    for i in range(num_steps):
        datetime_list.append(start_datetime + (offset + i) * increment)
    
    return datetime_list


def initialize(inFileName, forecastInFileName, startCol):

    print(inFileName)

    # load the new file
    # dataset = pd.read_csv(inFileName, header=0, infer_datetime_format=True, 
    #                         parse_dates=['UTC time'], index_col=['UTC time'])  # 17544 rows x 10 columns]    
    dataset = pd.read_csv(inFileName, header=0, parse_dates=['UTC time'], index_col=['UTC time'])  # 17544 rows x 10 columns]    
    
    # dataset = dataset[:8784]
    # print(dataset.head())  #  Unnamed: 0  carbon_intensity  coal  nat_gas  nuclear  oil  hydro  solar  wind  other
    # print(dataset.columns)
    dateTime = dataset.index.values

    # print(forecastInFileName)
    # forecastDataset = pd.read_csv(forecastInFileName, header=0, infer_datetime_format=True, 
    #                         parse_dates=['UTC time'], index_col=['UTC time'])     # [69888 rows x 15 columns]
    
    forecastDataset = pd.read_csv(forecastInFileName, header=0, parse_dates=['UTC time'], index_col=['UTC time'])     # [69888 rows x 15 columns]
    # dataset = dataset[:8784]
    # print(forecastDataset.head())
    # print(forecastDataset.columns)  
    # Index(['forecast_avg_wind_speed_wMean', 'forecast_avg_temperature_wMean',
    #    'forecast_avg_dewpoint_wMean', 'forecast_avg_dswrf_wMean',
    #    'forecast_avg_precipitation_wMean', 'avg_wind_production_forecast',
    #    'avg_solar_production_forecast', 'avg_nat_gas_production_forecast',
    #    'avg_coal_production_forecast', 'avg_nuclear_production_forecast',
    #    'avg_hydro_production_forecast', 'avg_oil_production_forecast',
    #    'avg_other_production_forecast', 'avg_demand_production_forecast',
    #    'avg_wind_production_forecast.1'],
    #   dtype='object')

    for i in range(startCol, len(dataset.columns.values)):
        col = dataset.columns.values[i]
        dataset[col] = dataset[col].astype(np.float64)
        # print(col, dataset[col].dtype)

    # print("\nAdding features related to date & time...")

    #TODO: convert utc time to local time, then add features
    modifiedDataset = addDateTimeFeatures(dataset, dateTime, startCol)
    dataset = modifiedDataset
    # print("Features related to date & time added")

    return dataset, forecastDataset, dateTime
    # return dataset, forecastDataset

# Date time feature engineering
def addDateTimeFeatures(dataset, dateTime, startCol):
    global DEPENDENT_VARIABLE_COL
    dates = []
    hourList = []
    hourSin, hourCos = [], []
    monthList = []
    monthSin, monthCos = [], []
    weekendList = []
    columns = dataset.columns
    secInDay = 24 * 60 * 60 # Seconds in day 
    secInYear = year = (365.25) * secInDay # Seconds in year 

    day = pd.to_datetime(dateTime[0])
    isWeekend = 0
    zero = 0
    one = 0
    for i in range(0, len(dateTime)):
        day = pd.to_datetime(dateTime[i])
        dates.append(day)
        hourList.append(day.hour)
        hourSin.append(np.sin(day.hour * (2 * np.pi / 24)))
        hourCos.append(np.cos(day.hour * (2 * np.pi / 24)))
        monthList.append(day.month)
        monthSin.append(np.sin(day.timestamp() * (2 * np.pi / secInYear)))
        monthCos.append(np.cos(day.timestamp() * (2 * np.pi / secInYear)))
        if (day.weekday() < 5):
            isWeekend = 0
            zero +=1
        else:
            isWeekend = 1
            one +=1
        weekendList.append(isWeekend)        
    loc = startCol+1
    # print(zero, one)
    # hour of day feature
    dataset.insert(loc=loc, column="hour_sin", value=hourSin)
    dataset.insert(loc=loc+1, column="hour_cos", value=hourCos)
    # month of year feature
    dataset.insert(loc=loc+2, column="month_sin", value=monthSin)
    dataset.insert(loc=loc+3, column="month_cos", value=monthCos)
    # is weekend feature
    dataset.insert(loc=loc+4, column="weekend", value=weekendList)

    # print(dataset.columns)
    # print(dataset.head())
    return dataset


def splitDataset(dataset, trainDataSize, testDataSize, seq_len): # testDataSize are in days
    # print("No. test days:", testDataSize)
    # print("No. of rows in dataset:", len(dataset))
    
    # valData = None
    # numTestEntries = testDataSize * 24
    n_train = int(trainDataSize * 24)
    n_val = 0
    n_test = int(testDataSize * 24)

    train_end = n_train
    val_end = n_train + n_val

    test_start = val_end - seq_len # # allow window to start earlier
    test_end = test_start + n_test + seq_len

    # dataset[start:end] eqaul to dataset[start:end, :]

    # trainData: everything except the last numTestEntries entries.
    # testData: the last numTestEntries entries — for testing.
    # trainData, testData = dataset[:-numTestEntries], dataset[-numTestEntries:]
    trainData, testData = dataset[0:train_end], dataset[test_start:test_end]

    # fullTrainData = np.copy(trainData)
    
    # New trainData: everything except the last numValEntries.
    # valData: the last numValEntries entries — used for validation during training.
    # trainData, valData = trainData[:-numValEntries], trainData[-numValEntries:]

    # trainData = trainData[:-predictionWindowDiff]
    # print("No. of rows in training set:", len(trainData))
    # print("No. of rows in validation set:", len(valData))
    # print("No. of rows in test set:", len(testData))

    # trainData: (8040, 10), valData: (720, 10), testData: (4368, 10), fullTrainData: (8760, 10)
    return trainData, testData


def splitWeatherDataset(dataset, testDataSize, predictionWindowHours): # testDataSize are in days

    print("No. of rows in weather dataset:", len(dataset))

    valData = None
    numTestEntries = testDataSize * predictionWindowHours # 17472
    # numValEntries = valDataSize * predictionWindowHours  # 2880

    trainData, testData = dataset[:-numTestEntries], dataset[-numTestEntries:]

    # fullTrainData = np.copy(trainData)
    # trainData, valData = trainData[:-numValEntries], trainData[-numValEntries:]
    print("No. of rows in training set:", len(trainData))  # 32160
    # print("No. of rows in validation set:", len(valData))  # 2880
    print("No. of rows in test set:", len(testData))       # 17472
    return trainData, testData

def splitWeatherDataset_v2(dataset, testDataSize): # testDataSize are in days

    # print("No. of rows in weather dataset:", len(dataset))

    valData = None
    numTestEntries = testDataSize * 24
    # numValEntries = valDataSize * predictionWindowHours  # 2880

    trainData, testData = dataset[:-numTestEntries], dataset[-numTestEntries:]

    # fullTrainData = np.copy(trainData)
    # trainData, valData = trainData[:-numValEntries], trainData[-numValEntries:]
    # print("No. of rows in training set:", len(trainData))  # 12264
    # print("No. of rows in validation set:", len(valData))  # 2880
    # print("No. of rows in test set:", len(testData))       # 5280
    return trainData, testData


def fillMissingData(data): # If some data is missing (NaN), use the same value as that of the previous row
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if(np.isnan(data[i, j])):
                data[i, j] = data[i-1, j]
    return data


def scaleDataset(trainData, testData):
    
    # Scaling columns to range (0, 1)
    row, col = trainData.shape[0], trainData.shape[1]
    ftMin, ftMax = [], []
    for i in range(col):
        fmax= trainData[0, i]
        fmin = trainData[0, i]
        for j in range(len(trainData[:, i])):
            if(fmax < trainData[j, i]):
                fmax = trainData[j, i]
            if (fmin > trainData[j, i]):
                fmin = trainData[j, i]
        ftMin.append(fmin)
        ftMax.append(fmax)
        # print(fmax, fmin)

    for i in range(col):
        if((ftMax[i] - ftMin[i]) == 0):
            continue
        trainData[:, i] = (trainData[:, i] - ftMin[i]) / (ftMax[i] - ftMin[i])
        # valData[:, i] = (valData[:, i] - ftMin[i]) / (ftMax[i] - ftMin[i])
        testData[:, i] = (testData[:, i] - ftMin[i]) / (ftMax[i] - ftMin[i])

    # return trainData, valData, testData, ftMin, ftMax
    return trainData, testData, ftMin, ftMax


def manipulateTrainingDataShape(data, carbon_index, trainWindowHours, labelWindowHours, weatherData=None):
    """
    sliding_windows is 1
    the output weatherX's time steps are the same as the input X's time steps
    """
    # trainWindowHours = 24
    # labelWindowHours = 96

    print("Data shape: ", data.shape)
    # global MAX_PREDICTION_WINDOW_HOURS
    # global PREDICTION_WINDOW_HOURS

    X, y, weatherX = list(), list(), list()
    weatherIdx = 0
    hourIdx = 0
    # step over the entire history one time step at a time
    for i in range(len(data) - (trainWindowHours + labelWindowHours) + 1):

        # define the end of the input sequence
        trainWindow = i + trainWindowHours
        labelWindow = trainWindow + labelWindowHours
        xInput = data[i: trainWindow, :]
        
        # xInput = xInput.reshape((len(xInput), 1))
        X.append(xInput)
        weatherX.append(weatherData[weatherIdx: weatherIdx + trainWindowHours])

        weatherIdx +=1
        hourIdx +=1
        if(hourIdx ==24):
            hourIdx = 0
            weatherIdx += (labelWindowHours - 24)

        y.append(data[trainWindow:labelWindow, carbon_index])

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    weatherX = np.array(weatherX, dtype=np.float64)
    # X = np.append(X, weatherX, axis=2)

    return X, weatherX, y


def manipulateTrainingDataShape_v2(data, carbon_index, trainWindowHours, labelWindowHours, sliding_windows=1, weatherData=None):
    """
    sliding_windows is 1
    the output weatherX's time steps are the same as the output Y's time steps
    """
    # trainWindowHours = 24
    # labelWindowHours = 96

    # print("Data shape: ", data.shape)
    # global MAX_PREDICTION_WINDOW_HOURS
    # global PREDICTION_WINDOW_HOURS

    X, y, weatherX = list(), list(), list()
    # weatherIdx = 0
    # hourIdx = 0
    # step over the entire history one time step at a time
    # for i in range(len(data) - (trainWindowHours + labelWindowHours) + 1):
    for i in range(0, len(data) - (trainWindowHours + labelWindowHours) + 1, sliding_windows):

        # define the end of the input sequence
        trainWindow = i + trainWindowHours
        labelWindow = trainWindow + labelWindowHours
        xInput = data[i: trainWindow, :]
        
        # xInput = xInput.reshape((len(xInput), 1))
        X.append(xInput)
        
        weather_input = weatherData[trainWindow: labelWindow]
        weatherX.append(weather_input)

        # weatherIdx +=1
        # hourIdx +=1
        # if(hourIdx ==24):
            # hourIdx = 0
            # weatherIdx += (labelWindowHours - 24)
        # weatherIdx += labelWindowHours

        y_one = data[trainWindow:labelWindow, carbon_index]
        y.append(y_one)

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    weatherX = np.array(weatherX, dtype=np.float64)
    # X = np.append(X, weatherX, axis=2)

    return X, weatherX, y


def inverseDataScaling(data, cmax, cmin):
    cdiff = cmax-cmin
    unscaledData = np.zeros_like(data)
    for i in range(data.shape[0]):
        unscaledData[i] = round(max(data[i] * cdiff + cmin, 0), 5)
    return unscaledData



if __name__ == '__main__':
    data_base_dir = '/kang/carbon/CarbonCast_main/data'
    region = 'CISO'
    # carbon_type = 'direct'
    carbon_type = 'lifecycle'

    data_base_dir_region = os.path.join(data_base_dir, region)

    # '/kang/carbon/CarbonCast_main/data/CISO/CISO_direct_emissions.csv'
    inFileName = os.path.join(data_base_dir_region, f'{region}_{carbon_type}_emissions.csv')
    
    # '/kang/carbon/CarbonCast_main/data/CISO/CISO_96hr_forecasts_DA.csv'
    # forecastInFileName = os.path.join(data_base_dir_region, f'{region}_96hr_forecasts_DA.csv')

    forecastInFileName = f"/mnt/ssd2/kangyang/carbon/TSFM_Building_carbon/data/weather/{region}_weather_2020_2021.csv"

    startCol = 1   # index in inFileName 
    number_of_features = 6 # carbon_intensity, plus 5 time-related features: (hour_sin, hour_cos, month_sin, month_cos, weekend)
    num_weather_features = 7 # in carboncast the valuse is 12, since it contains source prediction electricty from first tier model

    carbon_intensity_index = 0 # carbon_intensity index in dataset

    prediction_horizon_hours = 96

    trainWindowHours = 24
    labelWindowHours = prediction_horizon_hours
    sliding_windows = 1

    print("Initializing...")
    dataset, weatherDataset, dateTime = initialize(inFileName, forecastInFileName, startCol)
    print("***** Initialization done *****")

    time_step, feature_len = dataset.shape

    num_days = time_step // 24

    training_ratio = 0.7
    # testing_ratio = 1 - training_ratio

    num_train_days = int(num_days * training_ratio)
    num_test_days = num_days - num_train_days  # 220

    # split into train and test
    print("Spliting dataset into train/test...")
    # trainData: (12264, 6), testData: (5280, 6)
    trainData, testData = splitDataset(dataset.values, num_test_days)

    trainData = trainData[:, startCol: startCol + number_of_features]
    testData = testData[:, startCol: startCol + number_of_features]

    trainDates = dateTime[: -(num_test_days * 24):]
    testDates = dateTime[-(num_test_days * 24):]

    print("TrainData shape: ", trainData.shape) # days x hour x features (12264, 6)
    # print("ValData shape: ", valData.shape) # days x hour x features (4344, 6)
    print("TestData shape: ", testData.shape) # days x hour x features (5280, 6) splitWeatherDataset_v2

    # carbon_dataset_len = dataset.shape[0]
    weatherDataset = weatherDataset[: time_step]
    # wTrainData, wTestData = splitWeatherDataset(weatherDataset.values, num_test_days, prediction_horizon_hours)
    wTrainData, wTestData = splitWeatherDataset_v2(weatherDataset.values, num_test_days)

    wTrainData = wTrainData[:, :num_weather_features]
    # wValData = wValData[:, :numForecastFeatures]
    wTestData = wTestData[:, :num_weather_features]

    print("WeatherTrainData shape: ", wTrainData.shape) # (days x hour) x features (12264, 7)
    # print("WeatherValData shape: ", wValData.shape) # (days x hour) x features  (17376, 12)
    print("WeatherTestData shape: ", wTestData.shape) # (days x hour) x features  (5280, 7)

    trainData = fillMissingData(trainData)
    # valData = fillMissingData(valData)
    testData = fillMissingData(testData)

    wTrainData = fillMissingData(wTrainData)
    # wValData = fillMissingData(wValData)
    wTestData = fillMissingData(wTestData)

    print("***** Dataset split done *****")

    featureList = dataset.columns.values
    featureList = featureList[startCol:startCol+number_of_features].tolist()
    featureList.extend(weatherDataset.columns.values[:num_weather_features])
    print("Features: ", featureList)


    print("Scaling data...")

    # seems not used
    # unscaledTrainCarbonIntensity = np.zeros(trainData.shape[0])
    # for i in range(testData.shape[0]):
    #     unscaledTestData[i] = testData[i, DEPENDENT_VARIABLE_COL]
    # for i in range(trainData.shape[0]):
    #     unscaledTrainCarbonIntensity[i] = trainData[i, carbon_intensity_index]

    trainData, testData, ftMin, ftMax = scaleDataset(trainData, testData)
    print(trainData.shape, testData.shape)  # (12264, 6) (5280, 6)

    wTrainData, wTestData, wFtMin, wFtMax = scaleDataset(wTrainData, wTestData)
    print(wTrainData.shape, wTestData.shape)  # (12264, 7) (5280, 7)
    print("***** Data scaling done *****\n")

    # X, weatherX, y = manipulateTrainingDataShape(trainData, carbon_intensity_index, trainWindowHours, labelWindowHours, weatherData=wTrainData)

    X, weatherX, y = manipulateTrainingDataShape_v2(trainData, carbon_intensity_index, 
                                                    trainWindowHours, labelWindowHours, 
                                                    sliding_windows=sliding_windows, 
                                                    weatherData=wTrainData)
    # X = np.append(X, weatherX, axis=2)

    print("X shape: ", X.shape) # (days x hour) x features (12217, 24, 11)
    print("weatherX shape: ", weatherX.shape) # (days x hour) x features (12217, 24, 12)
    print("y shape: ", y.shape) # (days x hour) x features (12217, 96)

    # unscaledPredictedData = inverseDataScaling(predicted, ftMax[DEPENDENT_VARIABLE_COL], 
    #                     ftMin[DEPENDENT_VARIABLE_COL])
    
    unscaledTestData = inverseDataScaling(y, ftMax[carbon_intensity_index], 
                        ftMin[carbon_intensity_index])
