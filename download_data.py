from Historic_Crypto import Cryptocurrencies, HistoricalData, LiveCryptoData


data = HistoricalData('ETH-USD', 300, '2019-09-01-00-00').retrieve_data()

data.to_csv('ETH-USD-2020-06-01.csv', index=False)