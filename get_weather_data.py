import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry

import requests
import pandas as pd
from datetime import datetime
import os


def get_weather_data(latitude, longitude, start_date, end_date, variables):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": variables,
        # "timezone": "America/Los_Angeles"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(4).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(5).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["dew_point_2m"] = hourly_dew_point_2m
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["cloud_cover"] = hourly_cloud_cover

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    # print(hourly_dataframe)

    # Save to CSV
    # hourly_dataframe.to_csv("dswrf_32.0_-124.75_2019_2022_3.csv", index=False)

    return hourly_dataframe


def get_solar_radiation(lat, lon, years):

    # Settings
    # lat, lon = 32.0, -124.75
    base_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    # years = [2019, 2020, 2021, 2022]
    full_df = pd.DataFrame()

    # Loop through each year
    for year in years:
        # Define date range for each request
        start = f"{year}0101"
        end = f"{year}1231" if year != 2022 else "20220103"  # limit to Jan 3, 2022

        print(f"Fetching data: {start} to {end}...")

        # API request
        params = {
            "parameters": "ALLSKY_SFC_SW_DWN",
            "community": "RE",
            "longitude": lon,
            "latitude": lat,
            "start": start,
            "end": end,
            "format": "JSON"
        }
        response = requests.get(base_url, params=params)
        data = response.json()

        # Extract and format
        irradiance = data['properties']['parameter']['ALLSKY_SFC_SW_DWN']
        df = pd.DataFrame({
            # 'datetime': pd.to_datetime(list(irradiance.keys()), format='%Y%m%d:%H'),
            'datetime': pd.to_datetime(list(irradiance.keys()), format='%Y%m%d%H'),
            'DSWRF_W_m2': list(irradiance.values())
        })

        full_df = pd.concat([full_df, df])

    # Optional: sort and reset index
    full_df.sort_values(by='datetime', inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    # Preview
    print(full_df.head())
    # print(full_df.tail())

    # Save to CSV
    # full_df.to_csv("dswrf_32.0_-124.75_2019_2022.csv", index=False)
    return full_df


if __name__ == "__main__":

    outdir = "/kang/carbon/tsfm_carbon/data/weather/"
    os.makedirs(outdir, exist_ok=True)

    region_list = ["CISO", "PJM", "ERCO", "ISNE", "BPAT", "FPL", "NYISO", "SE", "DE", "PL", "ES", "NL", "AUS_QLD"]


    region_coordinates = {"CISO": (32.0, -124.75), 
                          "PJM": (34.25, -91.0), 
                          "ERCO": (25.25, -104.5), 
                          "ISNE": (40.0, -74.25), 
                            "BPAT": (39.5, -125.25),
                            "FPL": (24.0, -83.5),
                            "NYISO": (40.0, -80.25),
                          "SE": (55.25, 11.25), 
                          "DE": (47.25, 5.75),
                            "PL": (49.0, 14.0),
                            "ES": (36.0, -9.25),
                            "NL": (50.75, 3.25),
                            "AUS_QLD": (-29.75, 137.5)}
    region = "AUS_QLD"

    params_main = {
        "latitude": region_coordinates[region][0],
        "longitude": region_coordinates[region][1],
        # "latitude": 32.0,
        # "longitude": -124.75,
        # "start_date": "2019-01-01",
        # "end_date": "2022-01-03",
        "start_date": "2020-01-01",
        "end_date": "2021-12-31",
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "wind_speed_10m", "precipitation", "cloud_cover"],
        # "timezone": "America/Los_Angeles"
    }

    latitude_m = params_main["latitude"]
    longitude_m = params_main["longitude"]
    start_date_m = params_main["start_date"]
    end_date_m = params_main["end_date"]
    variables_m = params_main["hourly"]

    hourly_df = get_weather_data(latitude_m, longitude_m, start_date_m, end_date_m, variables_m)

    # hourly_df['date'] = hourly_df['date'].dt.tz_localize(None)

    # years = [2019, 2020, 2021, 2022]

    start_year = datetime.strptime(start_date_m, "%Y-%m-%d").year
    end_year = datetime.strptime(end_date_m, "%Y-%m-%d").year
    years = list(range(start_year, end_year + 1))

    solar_radiation_df = get_solar_radiation(latitude_m, longitude_m, years)
    solar_radiation_df['datetime'] = solar_radiation_df['datetime'].dt.tz_localize('UTC')

    # Merge datasets on datetime (UTC)
    combined_df = pd.merge(hourly_df, solar_radiation_df, how='inner', left_on='date', right_on='datetime')
    # combined_df['local_time'] = combined_df['date'].dt.tz_convert('America/Los_Angeles')

    # Optional: drop duplicate datetime column
    combined_df.drop(columns=['datetime'], inplace=True)
    combined_df = combined_df.rename(columns={"date": "UTC time"})

    # Save to CSV
    csv_path = os.path.join(outdir, f"{region}_weather_{years[0]}_{years[-1]}.csv")
    combined_df.to_csv(csv_path, index=False)

    # Preview
    print(combined_df.head())

