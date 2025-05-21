import numpy as np
import pandas as pd
from datetime import datetime
import pytz
from datetime import timedelta
from zoneinfo import ZoneInfo
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)


class Predictor:

    def __init__(self, location, time_zone, carbonIntensityColName, dateTimeCol, regions) -> None:
        self.carbonIntensityCol = carbonIntensityColName
        self.dateTimeCol = dateTimeCol
        self.loc = location
        self.timeZone = time_zone
        df = self.load_data()

        if (regions == "old"): # only 2020 & 2021 data to campare with CarbonCast
            df = df[df["year"] >= 2020] # OLD LOCATIONS
            df = df[df["year"] <= 2021] # OLD LOCATIONS
        
        dfs = []
        mape = []
        pred_df = pd.DataFrame()
        for hour in range(0, 24):
            trace = df[df["hour"]==hour]
            trace = trace.groupby(["year", "month", "day"]).mean(numeric_only=True).reset_index()
            trace = trace[["year", "month", "day", self.carbonIntensityCol]]
            trace[self.carbonIntensityCol].fillna(method="bfill", inplace=True)
            trace['prediction'] = trace[self.carbonIntensityCol].ewm(span=2, adjust=True).mean()
            trace['prediction'].fillna(method="bfill", inplace=True)
            trace["hour"] = hour
            trace["date"] = pd.to_datetime(trace.year*10000+trace.month*100+trace.day, format="%Y%m%d")
            dfs.append(trace)
            if (regions == "old"):
                trace1 = trace[trace["year"] == 2021] # OLD LOCATIONS
                trace1 = trace1[trace1["month"] > 6] # OLD LOCATIONS
            else:
                trace1 = trace[-439:] # NEW LOCATIONS
            mape.append(np.mean(np.abs((trace1[self.carbonIntensityCol].values[1:] - trace1["prediction"].values[:-1]) / trace1[self.carbonIntensityCol].values[1:])) * 100)
            pred_df = pd.concat([pred_df, trace1])

        print(self.loc, "day-ahead forecast", np.mean(mape))

        forecastCol =  pred_df["prediction"].values[:-1]
        pred_df = pred_df[1:]

        # print(forecastCol)
        # print(pred_df)
        pred_df["prediction"] = forecastCol
        pred_df.to_csv(f"./ewma-forecasts/{self.loc}_dayAheadForecasts.csv", index=False)

        self.trace = pd.concat(dfs).reset_index()

        return
    

    def load_data(self):
        df = pd.read_csv(f"./ewma-historical-ci-data/{self.loc}_lifecycle_emissions.csv")[[self.carbonIntensityCol, self.dateTimeCol]]
        tz = ZoneInfo(self.timeZone)

        if (self.loc != "DE"): # Fix Later
            df[self.dateTimeCol] = pd.to_datetime(df[self.dateTimeCol]).dt.tz_localize("UTC")
        else:
            df[self.dateTimeCol] = pd.to_datetime(df[self.dateTimeCol])

        df["datetime"] = df[self.dateTimeCol].dt.tz_convert(tz)
        df["month"] = df["datetime"].apply(lambda x: x.month)
        df["day"] = df["datetime"].apply(lambda x: x.day)
        df["hour"] = df["datetime"].apply(lambda x: x.hour)
        df["year"] = df["datetime"].apply(lambda x: x.year)
        df["weekday"] = df["datetime"].apply(lambda x: x.weekday)

        return df
    

    def getNDayForecasts(self, nDays):

        data = pd.read_csv(f"./ewma-forecasts/{self.loc}_dayAheadForecasts.csv")

        # Create overlapping windows of nDays rows
        window_size = nDays
        result = pd.concat([data.iloc[i: i + window_size] for i in range(len(data) - window_size + 1)], ignore_index=True)

        result[f"{nDays}_day_forecast"] = data['prediction'].repeat(nDays).reset_index(drop=True)
        result.drop(columns=["prediction"], inplace=True)
        result.to_csv(f"./ewma-forecasts/{self.loc}-{nDays}Dayforecasts.csv", index=False)

        mape = np.abs((result[carbonIntensityColName].values - result[f"{nDays}_day_forecast"].values) / result[carbonIntensityColName].values) * 100
        
        print(self.loc, "Mean", np.mean(mape))
        print(self.loc, "Median", np.percentile(mape, 50))
        print(self.loc, "90th percentile", np.percentile(mape, 90))
        print(self.loc, "95th percentile", np.percentile(mape, 95))

        return [np.mean(mape), np.percentile(mape, 50), np.percentile(mape, 90), np.percentile(mape, 95)]
    

    def getDaywiseForecasts(self, day, maxForecastHorizonDays=4):
        data = pd.read_csv(f"ewma-forecasts/{self.loc}-4Dayforecasts.csv")
        dayMape = []
        for i in range(day, len(data)-maxForecastHorizonDays, maxForecastHorizonDays):
            dayMape.append(np.abs(data[self.carbonIntensityCol].values[i] - data["4_day_forecast"].values[i])* 100 
                           / data[self.carbonIntensityCol].values[i])
        return dayMape
    
    
    
oldLocations = ["BPAT", "CISO", "ERCO", "FPL", "ISNE", "NYIS", "PJM", "DE", "ES", "NL", "PL", "SE", "AUS-QLD"]
newLocations = ["AR", "CL-SEN", "IN-NO", "IN-SO", "NZ", "UY"] # NEW LOCATIONS
allLocations = ["BPAT", "CISO", "ERCO", "FPL", "ISNE", "NYIS", "PJM", "DE", "ES", "NL", "PL", "SE", "AUS-QLD", 
                "AR", "CL-SEN", "IN-NO", "IN-SO", "NZ", "UY"]

allLocations = ["CISO"]
mapeLog = {}
if (not os.path.exists(f"./ewma-forecasts/")):
    os.makedirs(f"./ewma-forecasts")

for loc in allLocations:

    carbonIntensityColName = "carbon_intensity"
    dateTimeCol = "UTC time"
    regions = "old"
    
    if (loc in newLocations):
        carbonIntensityColName = "Carbon intensity gCO₂eq/kWh (Life cycle)" # NEW LOCATIONS
        dateTimeCol = "Datetime (UTC)" # NEW LOCATIONS
        regions = "new"

    print("Location: ", loc)

    p = Predictor(loc, "UTC", carbonIntensityColName, dateTimeCol, regions)
    
    print("4 day forecasts...")

    results = p.getNDayForecasts(nDays=4)
    mapeLog[loc] = results

#     print("Daywise forecasts...")
#     for i in range(4):
#         dayMape = p.getDaywiseForecasts(day=i, maxForecastHorizonDays=4)
#         print("Day ", i + 1)
#         print("Mean: ", np.mean(dayMape))
#         print("Median: ", np.percentile(dayMape, 50))
#         print("90th percentile: ", np.percentile(dayMape, 90))
#         print("95th percentile: ", np.percentile(dayMape, 95))

# df = pd.DataFrame.from_dict(mapeLog, orient='index')
# df.columns = ["Mean", "Median", "90th percentile", "95th percentile"]
# df.to_csv("./ewma-forecasts/4day-MAPE-percentile.csv")
    

