# Allora Data Fetcher

A script to fetch and process cryptocurrency data from the Tiingo API.

Attributes:

-tiingo_api_key (str): API key for Tiingo.

-symbol (str): Cryptocurrency symbol to fetch (default "ETHUSD").

-granularity (str): Time granularity for data points (e.g., "5min").

-start_date (datetime): Start date for the data.

-end_date (datetime): End date for the data (default: current UTC time).

-prediction_horizon (int): Prediction lag in minutes; determines return intervals.

-volatility_parameters (dict): Parameters for volatility calculation. Should include:
- 'volatility_granularity' (str): Granularity for volatility calculation (e.g., "1min").
-  'standardized' (bool): Whether to standardize volatility data.
 

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
