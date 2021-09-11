import re

import pandas as pd
from Historic_Crypto import Cryptocurrencies, HistoricalData, LiveCryptoData


parametres = [('BTC-USD', 300, '2019-09-01-00-00'),
              ('ETH-USD', 300, '2019-09-01-00-00'),
              ('LTC-USD', 300, '2019-09-01-00-00'),
              ('BTC-USD', 300, '2019-09-01-00-00'),
              ('ADA-USD', 300, '2019-09-01-00-00'),
              ('SOL-USD', 300, '2019-09-01-00-00'),
              ('XRP-USD', 300, '2019-09-01-00-00'),
              ('DOGE-USD', 300, '2019-09-01-00-00'),
              ('DOT-USD', 300, '2019-09-01-00-00'),
              ('UNI-USD', 300, '2019-09-01-00-00'),
              ('LUNA-USD', 300, '2019-09-01-00-00'),
              ('LINK-USD', 300, '2019-09-01-00-00'),
              ('ALGO-USD', 300, '2019-09-01-00-00'),
              ('AVAX-USD', 300, '2019-09-01-00-00'),
              ('ICP-USD', 300, '2019-09-01-00-00'),
              ('FTT-USD', 300, '2019-09-01-00-00'),
              ('MATIC-USD', 300, '2019-09-01-00-00'),
              ('FIL-USD', 300, '2019-09-01-00-00'),
              ]

data = []
for coin_pair, granularity, start_date in parametres:
    try:
        new_data = HistoricalData(coin_pair, granularity, start_date).retrieve_data()
        coin = re.sub(r'-USD$', '', coin_pair)
        new_data = new_data.rename({coin + '/' + col for col in new_data.columns}, axis=1)
        data.append(new_data)
    except:
        continue
from joblib import dump
with open('price_data', 'wb') as f:
    dump(data, f)
data = pd.concat(data, axis=1)
data.to_csv('Crypto-USD-2019-09-01-00-00.csv')
