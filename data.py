import os
import pandas as pd
import requests
from datetime import datetime, timedelta
import ta
from dune_client.client import DuneClient
import xml.etree.ElementTree as ET
import time
from pytrends.request import TrendReq
import yfinance as yf
import numpy as np
import holidays
import math
#from config import app_base_path, model_file_path, download_path // Use if you want to get a paths from your config.py
app_base_path = "/content"
app_base_path = os.getenv("APP_BASE_PATH", default=os.getcwd())
model_file_path = os.path.join(app_base_path, "models")
download_path = os.path.join(app_base_path, "data")

Dune_API_KEY   = "YOUR_DUNE_API_KEY"
FRED_API_KEY   = "YOUR_FRED_API_KEY"


class DataFetcher:

    def __init__(self, start_date, end_date= datetime.utcnow(), granularity = "D" , symbol = "ETHUSD", volatility = False, tiingo = "YOUR_TIINGO_API_KEY", cache_folder="data/sets", new=False):

        self.new = new
        self.tiingo = tiingo
        self.volatility = volatility
        self.symbol = symbol
        self.granularity = granularity
        self.cache_folder = cache_folder
        self.start_date = start_date
        self.end_date = end_date
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)  # Ensure the 'sets' folder exists


    def _generate_filename(self, symbol, frequency):
        """Generate a unique filename for the CSV based on the symbol and parameters."""
        start_date = self.start_date
        end_date = self.end_date
        return os.path.join(
            self.cache_folder, f"{symbol}_{start_date}_to_{end_date}_{frequency}.csv"
        )

    def _normalize_tiingo_data(self, data, asset_name):
        """Normalize Tiingo stock data to match the required schema."""
        if not data:
            print(f"No data available for {asset_name}")
            return pd.DataFrame()

        try:
            normalized_data = pd.DataFrame(data)
        except ValueError as e:
            print(f"Error in processing data for {asset_name}: {e}")
            return pd.DataFrame()

        normalized_data["date"] = pd.to_datetime(
            normalized_data["date"], errors="coerce"
        )

        return normalized_data[["date", "open", "high", "low", "close", "volume", "volumeNotional", "tradesDone"]]

    def fetch_tiingo_crypto_data(self, internal = False):
        """Fetch historical cryptocurrency data from Tiingo."""
        TIINGO_API_KEY = self.tiingo

        #map frequencys
        symbol = self.symbol
        granularity = self.granularity

        if granularity == "1440min":
          granularity = "1day"
        if granularity == "D":
          granularity = "1day"
        if granularity == "1h":
          granularity = "1hour"
        if granularity == "12h":
          granularity = "12hour"

        BASE_APIURL = "https://api.tiingo.com"
        filename = self._generate_filename(symbol, granularity)
        interval = 1401
        if internal == True:
          interval = 0
        cal_dic = {
            "1min" : (0, interval),
            "5min" : (1, 249),
            "10min" : (1, 105),
            "15min" : (1, 57),
            "30min" : (1, 9),
            "720min" : (20, 1),
            "1hour" : (2, 9),
            "12hour" : (20, 1),
            "1day" : (39, 0)
        }
        #25 + 14
        start_date = self.start_date.date()
        start_date = str(start_date)
        start_date = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=cal_dic[granularity][0])
        end_date = self.end_date

        # Define the URL, headers, and parameters for the request
        url = f"{BASE_APIURL}/tiingo/crypto/prices"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {TIINGO_API_KEY}",
        }

        params = {
            "tickers": symbol,
            "startDate": start_date,
            "endDate": end_date,
            "resampleFreq": granularity,
        }

        # Send request to Tiingo API
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return pd.DataFrame()

        # Parse JSON response
        try:
            data = response.json()
            if not data or "priceData" not in data[0]:
                print(f"No crypto data found for {symbol}")
                return pd.DataFrame()
        except (ValueError, KeyError) as e:
            print(f"Error parsing response data: {e}")
            return pd.DataFrame()

        df = self._normalize_tiingo_data(data[0]["priceData"], symbol)#.iloc[cal_dic[granularity][1]:]
        #print(cal_dic[granularity][1])

        df['Bollinger_High'] = ta.volatility.bollinger_hband(df['close'], window=20)
        df['Bollinger_Low'] = ta.volatility.bollinger_lband(df['close'], window=20)
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        df['MACD'] = ta.trend.macd(df['close'])
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)/ df['close']
        df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
        #turning into returns
        df["3_day_return"] = np.log(df["close"]/df["close"].shift(3))
        df["7_day_return"] = np.log(df["close"]/df["close"].shift(7))
        df["14_day_return"] = np.log(df["close"]/df["close"].shift(14))
        df["close_real"] = df["close"]
        cols = ["open", "high", "low", "close", "volume", 'Bollinger_High', 'Bollinger_Low', 'SMA_20', 'EMA_20']
        df[cols] = np.log(df[cols]/(df[cols].shift(1)))
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Save the fetched data to CSV
        #print(f"Saving crypto data to {filename}...")
        df.index.name = 'datetime'
        #df.to_csv(filename, index=False)
        #print(df)
        if internal == False:
          return df.loc[str(self.start_date):str(self.end_date)]
        if internal == True:
          return df

    def volatility_cal(self):
      save = self.granularity
      self.granularity = "1min"
      df = self.fetch_tiingo_crypto_data(internal = True)
      vol = pd.DataFrame([])
      df["volatility"] = df["close"].rolling(window=720).apply(np.std)
      df["volatility_6_hours"] = df["close"].rolling(window=360).apply(np.std)
      df["volatility_3_hours"] = df["close"].rolling(window=180).apply(np.std)
      df["volatility_1_hours"] = df["close"].rolling(window=60).apply(np.std)
      vol = df[(df.index.minute % 5 == 0) & (df.index.second == 0)]
      #print(vol)
      vol = vol.iloc[-576:]
      self.granularity = save
      return pd.DataFrame(vol[["volatility","volatility_6_hours"]]).loc[str(self.start_date):str(self.end_date)]


    def blockchain_info(self):
        granularity = self.granularity
        time = datetime.strptime(str(datetime.utcnow().date()), "%Y-%m-%d")
        save_path = os.path.join(download_path, f"dune-{time}.csv") #{granule}

        if os.path.exists(save_path) and self.new == False:
          df = pd.read_csv(save_path, index_col=0, parse_dates=True)
        else:

          dune = DuneClient(Dune_API_KEY)
          query_result = dune.get_latest_result(2899930)
          active_wallets = pd.DataFrame([query_result])
          active_wallets = pd.DataFrame(active_wallets["result"][0]["rows"])
          df = active_wallets
          df['datetime'] = pd.to_datetime(df['day'], errors='coerce')
          df = df.drop(columns=['volatility_30d', 'circulating', "day",'high_price','low_price',
                                            'ma100', 'ma200','open_price', 'sma100', 'sma200','total_fees',
                                            'total_mint','volatility_200d', 'volatility_90d', 'volume', 'volume_in_dollar', 'miner_revenue'])
          if df['datetime'].dt.tz is None:
              df['datetime'] = df['datetime'].dt.tz_localize('UTC', ambiguous='NaT')
          else:
              df['datetime'] = df['datetime'].dt.tz_convert('UTC')
          df.set_index('datetime', inplace=True)
          df[df.columns] = np.log(df[df.columns]/(df[df.columns].shift(1)))
          df.to_csv(save_path)

        df.index = pd.to_datetime(df.index)
        full_date_range = pd.date_range(start=df.index.min(), end=pd.Timestamp.today(tz='UTC'), freq=granularity)
        df = df.reindex(full_date_range)
        df = df.ffill().bfill()
        df = df.reset_index().rename(columns={'index': 'datetime'})
        df.set_index('datetime', inplace=True)
        df.index.name = 'datetime'
        #print(df)
        return df.loc[str(self.start_date):str(self.end_date)]


    def fetch_oecd_data(self, urls, max_retries=5):
        granularity = self.granularity
        all_data = pd.DataFrame([])

        time = datetime.strptime(str(datetime.utcnow().date()), "%Y-%m-%d")
        save_path = os.path.join(download_path, f"oecd-{time}.csv")
        try:
          if os.path.exists(save_path) and self.new == False:
            all_data = pd.read_csv(save_path, index_col=0, parse_dates=True)
          else:
              for key, url in urls.items():
                  retries = 0
                  while retries < max_retries:
                      try:
                          response = requests.get(url)
                          if response.status_code == 200:
                              root = ET.fromstring(response.content)
                              dates = []
                              values = []
                              namespaces = {
                                  'ns0': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message',
                                  'ns1': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common',
                                  'ns2': 'http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic'
                              }
                              for series in root.findall('.//ns2:Series', namespaces):
                                  for obs in series.findall('.//ns2:Obs', namespaces):
                                      date = obs.find('.//ns2:ObsDimension', namespaces).attrib['value']
                                      value = obs.find('.//ns2:ObsValue', namespaces).attrib['value']
                                      dates.append(date)
                                      values.append(value)
                              data = pd.DataFrame({'date': dates, 'Value': values})
                              data['date'] = pd.to_datetime(data['date'], format='%Y-%m')
                              data = data.sort_values(by='date').reset_index(drop=True)
                              data['Value'] = pd.to_numeric(data['Value'])
                              data['Value'] = data["Value"].diff().dropna()
                              today = datetime.utcnow().date()
                              data = data.set_index('date').reindex(pd.date_range(start=data['date'].min(), end=today, freq='D'))
                              data['Value'] = data['Value'].ffill().bfill()
                              all_data[key] = data

                              break
                          elif response.status_code == 429:
                              retries += 1
                              print(f"Rate limit exceeded. Retrying in 30 seconds...")
                              time.sleep(60)
                          else:
                              print(f"Failed to download XML data from {url}. Status code: {response.status_code}")
                              break
                      except Exception as e:
                          print(f"An error occurred while processing {url}: {e}")
                          break
              all_data.index.name = 'datetime'

              all_data.to_csv(save_path)

          all_data.index = pd.to_datetime(all_data.index)
          full_date_range = pd.date_range(start=all_data.index.min(), end=pd.Timestamp.today(), freq=granularity)
          all_data = all_data.reindex(full_date_range)
          all_data = all_data.ffill().bfill()
          #print(all_data)
          return all_data[self.start_date: self.end_date]
        except Exception as e:
          print(f"An error occurred: {e}")
          return None

    def fetch_data_for_series(self, series_id, value):

        url = f'https://api.stlouisfed.org/fred/series/observations?series_id={value}&api_key={FRED_API_KEY}&file_type=json'
        response = requests.get(url)
        data = response.json()
        observations = data['observations']
        df = pd.DataFrame(observations)
        df['date'] = pd.to_datetime(df['date'])
        df[series_id] = pd.to_numeric(df['value'], errors='coerce')
        df.set_index('date', inplace=True)
        df = df.drop(columns=["realtime_start", "realtime_end", "value"])
        #print(df)
        return df

    def resample_to_daily(self, df, series_id, end_date):

        df = df.reindex(pd.date_range(start=df.index.min(), end=end_date, freq='D'))
        df_daily = df.resample('D').asfreq()
        df_daily = df_daily.ffill().bfill()

        return df_daily

    def get_fred_data(self, dic_stlouis):
      granularity = self.granularity
      all_data = pd.DataFrame([])
      today = datetime.utcnow().date()

      time = datetime.strptime(str(datetime.utcnow().date()), "%Y-%m-%d")
      save_path = os.path.join(download_path, f"stlouis-{time}.csv")
      if os.path.exists(save_path) and self.new == False:
        all_data = pd.read_csv(save_path, index_col=0, parse_dates=True)
      else:
        for series_id, value in dic_stlouis.items():
            print(f"Fetching data for series: {series_id}")
            df_temp = pd.DataFrame([])
            df = self.fetch_data_for_series(series_id, value)
            df_temp = self.resample_to_daily(df, series_id, today)

            # Concatenate the data into the all_data DataFrame
            if all_data.empty:
                all_data = df_temp
            else:
                all_data = all_data.join(df_temp, how='outer')
        all_data = all_data.dropna()
        all_data.index.name = 'datetime'
        all_data.to_csv(save_path)

      all_data.index = pd.to_datetime(all_data.index)
      full_date_range = pd.date_range(start=all_data.index.min(), end=pd.Timestamp.today(), freq=granularity)
      all_data = all_data.reindex(full_date_range)
      all_data = all_data.ffill().bfill()
      #print(all_data)
      return all_data.loc[self.start_date: self.end_date]


    def fetch_google_trends_csv(self, keywords, geo='', category=0, gprop=''):
        granularity = self.granularity


        # Set pandas option to avoid future warnings
        pd.set_option('future.no_silent_downcasting', True)
        try:
            time = datetime.strptime(str(datetime.utcnow().date()), "%Y-%m-%d")
            save_path = os.path.join(download_path, f"google-{granularity}-{time}.csv") #{granule}
            if os.path.exists(save_path) and self.new == False:
              data = pd.read_csv(save_path, index_col=0, parse_dates=True)
              #print(data)
            else:
              tries = 0
              while tries < 4:
                try:
                  pytrends = TrendReq(hl='en-US', tz=360)
                  pytrends.build_payload(keywords, cat=category, timeframe=f'{self.start_date.date()} {self.end_date.date()}', geo=geo, gprop=gprop)
                  data = pytrends.interest_over_time()
                  if data.empty:
                      print("No data found for the given parameters.")
                      return
                  if 'isPartial' in data.columns:
                      data = data.drop(columns=['isPartial'])
                  tries = 5
                except:
                  tries +=1
                  time.sleep(4)

              data.index.name = 'datetime'
              data.to_csv(save_path)
            data.index = pd.to_datetime(data.index)
            full_date_range = pd.date_range(start=data.index.min(), end=pd.Timestamp.today(), freq=granularity)
            data = data.reindex(full_date_range)
            data = data.ffill().bfill()
            return data.loc[self.start_date: self.end_date]
        except Exception as e:
            print(f"An error occurred: {e}")

    def download_and_concat_yahoo_finance_data(self, tickers, filename='yahoo_finance_data.csv'):
        granularity = self.granularity
        start_date = self.start_date - timedelta(days=730)
        date_range = pd.date_range(start=str(str(start_date.date())), end=(str(self.end_date.date())))
        all_data = pd.DataFrame(index=date_range)
        time = datetime.strptime(str(datetime.utcnow().date()), "%Y-%m-%d")
        save_path = os.path.join(download_path, f"yahoo-{time}.csv")

        if os.path.exists(save_path) and self.new == False:
          all_data = pd.read_csv(save_path, index_col=0, parse_dates=True)
          #print(all_data)
        else:
            for ticker in tickers:
                try:
                    data = yf.download(ticker, start=start_date.date(), end=self.end_date.date())
                    if data.empty:
                        print(f"No data found for {ticker}.")
                        continue
                    all_data[ticker] = np.log(data["Close"]/data["Close"].shift(1))
                except Exception as e:
                    print(f"An error occurred for {ticker}: {e}")
            #print(all_data)
            all_data.index.name = 'datetime'
            all_data.to_csv(save_path)


        all_data.index = pd.to_datetime(all_data.index)
        full_date_range = pd.date_range(start=all_data.index.min(), end=pd.Timestamp.today(), freq=granularity)
        all_data = all_data.reindex(full_date_range)
        all_data = all_data.ffill().bfill()
        #print(all_data)
        return all_data.loc[self.start_date: self.end_date]

    def train_vol(self, symbol, urls_oecd, dic_stlouis, keywords, tickers, train = False):


      time = datetime.strptime(str(datetime.utcnow().date()), "%Y-%m-%d")
      save_path = os.path.join(download_path, f"yahoo-{time}.csv") #{granule}
      #print(save_path)

      if os.path.exists(save_path) == False:

        diff = self.end_date - self.start_date
        save = self.start_date
        if int(diff.days) < 7 :
          self.start_date = self.end_date - timedelta(days= 8)

      #data_dune = self.blockchain_info()
      data_google = self.fetch_google_trends_csv(keywords)

      data_oecd = self.fetch_oecd_data(urls_oecd)
      data_stlouis = self.get_fred_data(dic_stlouis)
      data_yahoo = self.download_and_concat_yahoo_finance_data(tickers)

      days_training = self.end_date - self.start_date
      days_training = int(days_training.days / 2)

      training_sum_volaitility = pd.DataFrame([])

      if self.volatility == True:
        self.granularity = "5min"
        for interval in reversed(range(days_training)):
          print(interval)
          self.start_date = datetime.utcnow() - timedelta(days= (interval+1)*2)
          self.end_date = datetime.utcnow() - timedelta(days= (interval)*2)

          training = self.fetch_tiingo_crypto_data()
          training[["volatility","volatility_6_hours"]] = self.volatility_cal().values
          training_sum_volaitility = pd.concat([training_sum_volaitility, training])
          #print(training_sum_volaitility)
        if train == True:
          training_sum_volaitility["volatility_target"] = training_sum_volaitility["volatility"].shift(-144)

      else:
        for interval in reversed(range(days_training)):
          self.start_date = datetime.utcnow() - timedelta(days= (interval+1)*2)
          self.end_date = datetime.utcnow() - timedelta(days= (interval)*2)
          data_tiingo = self.fetch_tiingo_crypto_data()
          training = data_tiingo
          training_sum_volaitility = pd.concat([training_sum_volaitility, training])

      data_list = [training_sum_volaitility, data_yahoo,data_stlouis,  data_google]#, data_dune , data_oecd
      for i, data in enumerate(data_list):

              if 'date' in data.columns:
                  data['date'] = pd.to_datetime(data['date'])
                  data.set_index('date', inplace=True)
              if not isinstance(data.index, pd.DatetimeIndex):
                  data.index = pd.to_datetime(data.index)
              if data.index.tz is None:
                  data.index = data.index.tz_localize('UTC')
              else:
                  data.index = data.index.tz_convert('UTC')

              # Sort the index
              data.sort_index(inplace=True)

              data_list[i] = data
      df = pd.concat(data_list, axis=1, join='inner').dropna()
      df.index.name = 'datetime'

      df['hour'] = df.index.hour
      df['minute'] = df.index.minute
      df['second'] = df.index.second
      df['day_of_week'] = df.index.dayofweek
      df['second_of_day'] = (df['hour'] * 3600) + (df['minute'] * 60) + df['second']
      df['second_of_day_sin'] = np.sin(2 * np.pi * df['second_of_day'] / 86400)
      df['second_of_day_cos'] = np.cos(2 * np.pi * df['second_of_day'] / 86400)
      df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
      df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
      us_holidays = holidays.US()
      df['is_holiday'] = [date in us_holidays for date in df.index.date]
      df = df.drop(columns=["hour","minute", "second", "day_of_week", "second_of_day","tradesDone","SMA_20","EMA_20"])
      if self.volatility == True:
        df["lagged_0.25_volatility"] = df["volatility"].shift(36)
        df["lagged_0.5_volatility"] = df["volatility"].shift(72)
        df["lagged_1_volatility"] = df["volatility"].shift(144)
        df["lagged_2_volatility"] = df["volatility"].shift(144*2)
        df["lagged_3_volatility"] = df["volatility"].shift(144*3)

      df = df.dropna()
      return df

######STILL IN PROGRESS######
    def data_ensemble(self):
      tiingo = "c30101f1eeaa985075b3b04c352e7fbd2956bd59"#"afc3e8df6c211ee2f12b33775ba6d199e3efc4cd"#
      xgb_data = self.train_vol(symbol = "ETHUSD", urls_oecd = urls_oecd, dic_stlouis = dic_stlouis, keywords= keywords, tickers = tickers ,train = True)
      self.start_date = self.start_date - timedelta(days=180)
      self.end_date = datetime.utcnow()
      self.granularity = "720min"
      garch_data = self.fetch_tiingo_crypto_data()
      ensemble = xgb_data
      ensemble["MLP_data"] = [np.concatenate([garch_data.loc[:xgb_data.index[i]].iloc[int(np.floor(i/144)):179+int(np.floor(i/144))]["close"].values, [xgb_data["close"][i]]]) for i in range(len(xgb_data))]
      return ensemble#ensemble, garch_data
