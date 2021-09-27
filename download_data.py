import re
from joblib import dump
import pandas as pd
from Historic_Crypto import Cryptocurrencies, HistoricalData, LiveCryptoData


# parameters = [('BTC-USD', 300, '2019-09-01-00-00'),
#               ('ETH-USD', 300, '2019-09-01-00-00'),
#               ('LTC-USD', 300, '2019-09-01-00-00'),
#               ('ADA-USD', 300, '2019-09-01-00-00'),
#               ('SOL-USD', 300, '2019-09-01-00-00'),
#               ('XRP-USD', 300, '2019-09-01-00-00'),
#               ('DOGE-USD', 300, '2019-09-01-00-00'),
#               ('DOT-USD', 300, '2019-09-01-00-00'),
#               ('UNI-USD', 300, '2019-09-01-00-00'),
#               ('LUNA-USD', 300, '2019-09-01-00-00'),
#               ('LINK-USD', 300, '2019-09-01-00-00'),
#               ('ALGO-USD', 300, '2019-09-01-00-00'),
#               ('AVAX-USD', 300, '2019-09-01-00-00'),
#               ('ICP-USD', 300, '2019-09-01-00-00'),
#               ('FTT-USD', 300, '2019-09-01-00-00'),
#               ('MATIC-USD', 300, '2019-09-01-00-00'),
#               ('FIL-USD', 300, '2019-09-01-00-00'),
#               ]
parameters = [('BTC-USD', 300, '2017-01-01-00-00'),
              ('ETH-USD', 300, '2017-01-01-00-00'),
              ('LTC-USD', 300, '2017-01-01-00-00'),
              ('LINK-USD', 300, '2017-01-01-00-00'),
              ('ALGO-USD', 300, '2017-01-01-00-00'),
              ('BNB-USD', 300, '2017-01-01-00-00'),
              ('DOGE-USD', 300, '2017-01-01-00-00'),
              ('ADA-USD', 300, '2017-01-01-00-00'),
              ('XRP-USD', 300, '2017-01-01-00-00'),
              ('NEO-USD', 300, '2017-01-01-00-00'),
              ('OMG-USD', 300, '2017-01-01-00-00'),
              ]
data = []
for coin_pair, granularity, start_date in parameters:
    try:
        new_data = HistoricalData(coin_pair, granularity, start_date).retrieve_data()
        coin = re.sub(r'-USD$', '', coin_pair)
        new_data = new_data.rename({col: coin + '/' + col for col in new_data.columns}, axis=1)
        data.append(new_data)
    except:
        pass

with open('crypto_data/price_data', 'wb') as f:
    dump(data, f)
data = pd.concat(data, axis=1)

# Multi-index by coin, category
coins = [re.findall('^(\w+)/', col)[0] for col in data.columns]
category = [re.findall(r'/(\w+)$', col)[0] for col in data.columns]
data.columns = pd.MultiIndex.from_tuples(list(zip(category, coins)), names=['category', 'coin'])
data = data.sort_values(by=['category', 'coin'], axis=1)
data.to_csv('crypto_data/Crypto-USD-2017-01-01.csv')
