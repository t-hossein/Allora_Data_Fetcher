# Allora Data Fetcher

Allora Data Fetcher is a Python tool designed to simplify the process of creating robust cryptocurrency datasets for predictive modeling. It extends beyond the raw data provided by Tiingo by automatically adjusting key parameters such as data granularity and prediction horizon. Specifically, it:

- Aggregates Data Appropriately: Converts high-frequency (e.g., 5-minute) data into intervals that match the prediction horizon (e.g., 6-hour), summing cumulative values like volume and computing the maximum or minimum for high/low values.

- Preprocesses Data: Detects and interpolates missing values to ensure a clean dataset.

- Extracts Features: Generates a comprehensive feature set—including technical indicators like RSI, MACD, and EMA, as well as custom lag returns and seasonal components—to facilitate effective model training.

- Handles Flexible Parameters: Allows users to easily adjust parameters such as the start/end date, symbol, granularity, and prediction horizon. It also supports optional volatility calculations using log returns.

This streamlined approach helps overcome limitations of raw Tiingo data (like the 5000 data point per request cap) and ensures that your dataset is well-tailored for building accurate cryptocurrency price prediction models.

![image](https://github.com/user-attachments/assets/fdd899ba-349f-4f6f-b230-42452040e808)
![image](https://github.com/user-attachments/assets/3363ef5d-f392-4d72-a5bb-fe2244e484cb)


See the following Google Colab notebook as an example: ----In development----

Attributes:

- tiingo_api_key (str): API key for Tiingo.
- symbol (str): Cryptocurrency symbol to fetch (default "ETHUSD").
- granularity (str): Time granularity for data points (e.g., "5min").
- start_date (datetime): Start date for the data.
- end_date (datetime): End date for the data (default: current UTC time).
- prediction_horizon (int): Prediction lag in minutes; determines return intervals.
- volatility_parameters (dict): Parameters for volatility calculation. Should include:
   -  'volatility_granularity' (str): Granularity for volatility calculation (e.g., "1min").
   -   'standardized' (bool): Whether to standardize volatility data.
 

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




