import json
from flask import Flask, Response
#from config import app_base_path, model_file_path, download_path
import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, Response
import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
import traceback
import logging
import time
import data
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib


urls_oecd = {
    "Growth Rate": "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_KEI@DF_KEI,4.0/USA.M.PRVM.IX.BTE..?startPeriod=2020-01",
    "Unemployment Rate": "https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,1.0/USA..._Z.Y._T.Y_GE15..M?startPeriod=2008-01"
}

dic_stlouis = {
    "CPI_1":"MEDCPIM158SFRBCLE",
    "Dollar Index":'DTWEXBGS',
    "Uncertainty":'VIXCLS',
    "CPI_2":"CORESTICKM159SFRBATL"
    }


tickers = ['USCI', '^KS11', '000300.SS', "GC=F", "CL=F", "ES=F", "DX-Y.NYB", "EURUSD=X"]#

keywords = ['ETH', "BTC"]


granule = 5
start_date = datetime.utcnow() - timedelta(days=4)
end_date = datetime.utcnow()
tiingo = "YOUR_TIINGO_API_KEY"
dataobj = DataFetcher(start_date = start_date, end_date = end_date,symbol = "BTCUSD", granularity = f"{granule}min", volatility = False, tiingo=tiingo, new =True)
train = dataobj.train_vol(symbol = "BTCUSD", urls_oecd= urls_oecd, dic_stlouis = dic_stlouis, keywords= keywords, tickers = tickers, train = False)
