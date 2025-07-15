from data import DataFetcher
from datetime import datetime, timedelta
start_date=datetime.utcnow() - timedelta(days=20)
end_date=datetime.utcnow()
TIINGO_API_KEY = "YOUR_API_TIINGO_KEY"
# volatility_parameters = {"volatility_granularity": "5min", "standardized":True}
data = DataFetcher(start_date = start_date, end_date = end_date ,symbol = "paxgusd", granularity = "5min", TIINGO_API_KEY=TIINGO_API_KEY)
price_data = data.tiingo_crypto_data(prediction_horizon=1440)
