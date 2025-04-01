import os
import re
import time
import math
import requests
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import ta
from statsmodels.tsa.seasonal import seasonal_decompose

class DataFetcher:
    """
    Class to fetch and process cryptocurrency data from the Tiingo API.

    Attributes:
        tiingo_api_key (str): API key for Tiingo.
        symbol (str): Cryptocurrency symbol to fetch (default "ETHUSD").
        granularity (str): Time granularity for data points (e.g., "5min").
        start_date (datetime): Start date for the data.
        end_date (datetime): End date for the data (default: current UTC time).
    """

    def __init__(self, start_date, end_date=datetime.utcnow(), granularity="5min",
                 symbol="ETHUSD", TIINGO_API_KEY=None):
        """
        Initialize DataFetcher with date range, granularity, symbol and API key.
        """
        self.tiingo_api_key = TIINGO_API_KEY
        self.symbol = symbol
        self.granularity = granularity
        self.start_date = start_date
        self.end_date = end_date

    def extract_time_components(self, granularity):
        """
        Extract the numerical value and unit from a granularity string.

        Parameters:
            granularity (str): A string representing the granularity, e.g., "5min", "1h", "1D".

        Returns:
            tuple: (number, unit) where unit is standardized (e.g., minutes or day).

        Notes:
            - If the unit is hours ("h"), converts to minutes.
            - If the granularity in minutes is exactly 1440, it is converted to days.
        """
        match = re.match(r"(\d+)([a-zA-Z]+)", granularity)
        if not match:
            raise ValueError("Granularity format not recognized. Expected format like '5min', '1h', '1D'.")
        number = int(match.group(1))
        unit = match.group(2)

        # Convert hours to minutes if necessary
        if unit == "h":
            unit = "min"
            number *= 60
        # Convert 1440 minutes to 1 day (Tiingo issues 1440-min intervals)
        elif unit == "min" and number % 1440 == 0:
            number //= 1440
            unit = "day"
        elif unit == "D":
            unit = "day"

        return number, unit

    def tiingo_crypto_data(self, prediction_horizon=5, volatility_parameters=None):
        """
        Fetch cryptocurrency data from Tiingo API and compute technical indicators.

        Parameters:
            prediction_horizon (int): Prediction lag in minutes; determines return intervals.
            volatility_parameters (dict): Parameters for volatility calculation. Should include:
                - 'volatility_granularity' (str): Granularity for volatility calculation (e.g., "1min").
                - 'standardized' (bool): Whether to standardize volatility data.

        Returns:
            pd.DataFrame: Processed DataFrame with technical indicators and features.

        Process Overview:
            1. Extract time components from granularity and (if provided) volatility granularity.
            2. Fetch data using the Tiingo API for the given symbol.
            3. Extend data fetch if the initial request does not cover the desired start date.
            4. Reindex and interpolate missing timestamps.
            5. Compute volatility metrics if requested.
            6. Aggregate data to match the desired time frame.
            7. Calculate various technical analysis indicators (e.g., Bollinger bands, RSI, MACD, ATR, etc.).
            8. Compute returns and lagged returns.
            9. Add seasonality features.
        """
        # Validate and extract granularity components
        G_number, G_unit = self.extract_time_components(self.granularity)
        lags = range(2, 31)  # Lags for computing lagged return features

        # Setup volatility parameters if provided
        VG_number = G_number
        standard_factor = 1  # Default standard factor
        if volatility_parameters:
            volatility_granularity = volatility_parameters["volatility_granularity"]
            standardized = volatility_parameters["standardized"]
            VG_number, VG_unit = self.extract_time_components(volatility_granularity)
            if G_number < VG_number:
                raise Exception("Granularity must be equal or larger than volatility granularity.")

        # Construct the granularity string for the API request
        granularity_str = f"{VG_number}{G_unit}"
        window = int(prediction_horizon / VG_number)  # Window length for rolling calculations

        BASE_APIURL = "https://api.tiingo.com"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.tiingo_api_key}",
        }
        params = {
            "tickers": self.symbol,
            "endDate": str(self.end_date),
            "resampleFreq": granularity_str,
        }
        url = f"{BASE_APIURL}/tiingo/crypto/prices"

        # Initial API request
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Initial API request error: {e}")
            return pd.DataFrame()

        try:
            json_data = response.json()
            # Check if data exists and the key 'priceData' is available
            if not json_data or "priceData" not in json_data[0]:
                print(f"No crypto data found for {self.symbol}")
                return pd.DataFrame()
        except (ValueError, KeyError) as e:
            print(f"Error parsing response data: {e}")
            return pd.DataFrame()

        # Convert API JSON to DataFrame
        df = pd.DataFrame(json_data[0]['priceData'])
        # Set as index
        df.set_index("date", inplace=True)

        # Ensure the start_date is covered (Tiingo returns a limited number of datapoints)
        extend_points = window * (max(lags) if lags else 1) + 100 * int(G_number / VG_number)
        # Loop to extend data if the earliest fetched data is later than required start_date
        while  not (df.index[extend_points] <= self.start_date.isoformat() <= df.index[-1]):
            # Update endDate to the earliest timestamp fetched
            params["endDate"] = str(df.index[0])
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Extended API request error: {e}")
                break

            temp_df = pd.DataFrame(response.json()[0]['priceData'])
            temp_df.set_index("date", inplace=True)
            # Concatenate new data with existing data and remove duplicate indices
            df = pd.concat([temp_df, df]).sort_index()
            df = df[~df.index.duplicated(keep='first')]

        # Rename index for clarity and convert index to datetime and
        df.index = pd.to_datetime(df.index)
        df.index.name = 'datetime'
        # Reindex DataFrame to ensure consistent timestamps
        df = df.asfreq(granularity_str)
        missing_timestamps = df[df.isnull().any(axis=1)].index
        if not missing_timestamps.empty:
            print("Number of missing timestamps:", len(missing_timestamps))
            # Interpolate missing values using time-based method
            df = df.interpolate(method='time')

        # Compute volatility features if parameters provided
        if volatility_parameters:
            if standardized:
                standard_factor = np.sqrt(window)
            df["volatility"] = df["close"].rolling(window=window).std() * standard_factor
            df["volatility_half"] = df["close"].rolling(window=max(1, int(window / 2))).std() * standard_factor
            df["volatility_quartet"] = df["close"].rolling(window=max(1, int(window / 4))).std() * standard_factor

        # Adjust aggregation to match the prediction horizon
        df[["volume", "volumeNotional", "tradesDone"]] = df[["volume", "volumeNotional", "tradesDone"]].rolling(window=window).sum()
        df["high"] = df["high"].rolling(window=window).max()
        df["low"] = df["low"].rolling(window=window).min()

        # Filter rows to ensure indices align with the expected granularity (assumes G_unit in minutes)
        df = df[(df.index.minute % G_number == 0) & (df.index.second == 0)]
        # Compute technical analysis indicators using the `ta` library
        df['Bollinger_High'] = ta.volatility.bollinger_hband(df['close'], window=20)
        df['Bollinger_Low'] = ta.volatility.bollinger_lband(df['close'], window=20)
        df['RSI_10'] = ta.momentum.rsi(df['close'], window=10)
        df['RSI_100'] = ta.momentum.rsi(df['close'], window=100)
        df['MACD'] = ta.trend.macd(df['close'])
        if int(prediction_horizon/G_number) <= 14: # If long prediction horizon and short granularities are used, errors will occur
          df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14) / df['close']
        df['KST'] = ta.trend.kst(df['close'])
        df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_100'] = ta.trend.sma_indicator(df['close'], window=100)
        df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
        df["EMA_100"] = ta.trend.ema_indicator(df['close'], window=100)
        df["std_0.05"] = df["close"].ewm(alpha=0.05).std()
        df["std_0.1"] = df["close"].ewm(alpha=0.1).std()
        df["diff_trend"] = np.log(df["close"]/df['EMA_20'].shift(+1))
        df["high-low"] = df["high"] - df["low"]
        df["high-open"] = df["high"] - df["open"]
        df["low-open"] = df["low"] - df["open"]
        df["close-open"] = df["close"] - df["open"]

        # Calculate returns and adjusted returns based on the defined window
        df["return_open"] = df["open"]
        df["return"] = df["close"]
        df["log_volume"] = df["volume"]
        cols = ["return_open", "return", "log_volume", "SMA_20", "EMA_20"]
        df[cols] = np.log(df[cols] / df[cols].shift(window))
        df["open-close_return"] = df["return_open"] - df["return"]

        # Compute lagged returns for additional features
        for lag in lags:
            df[f"{lag}_lag_return"] = np.log(df["close"] / df["close"].shift(lag * window))

        # Drop any rows with missing values
        df.dropna(inplace=True)

        # Add seasonality features via decomposition
        decomposition = seasonal_decompose(df["return"], period=72, model="additive", extrapolate_trend="freq")
        df["seasonal_decomposition"] = decomposition.seasonal

        # Generate cyclic time features based on time of day
        df["hour"] = df.index.hour
        df["minute"] = df.index.minute
        df["second"] = df.index.second
        df["second_of_day"] = (df["hour"] * 3600) + (df["minute"] * 60) + df["second"]
        df["second_of_day_sin"] = np.sin(2 * np.pi * df["second_of_day"] / 86400)
        df["second_of_day_cos"] = np.cos(2 * np.pi * df["second_of_day"] / 86400)
        # Drop intermediate columns used for cyclic features
        df.drop(columns=["hour", "minute", "second", "second_of_day"], inplace=True)

        # Restrict the final DataFrame to the desired date interval
        df = df.loc[str(self.start_date):str(self.end_date)]
        return df
