import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from preprocessing import TimeSeriesImputer

df = pd.read_csv('crypto_data/Crypto-USD-2017-01-01.csv', index_col='time')
df.index = pd.to_datetime(df.index)

first_valid_idx = max([df[col].first_valid_index() for col in df.columns])
last_valid_idx = min([df[col].last_valid_index() for col in df.columns])
columns_of_interest = [col for col in df.columns if 'close' in col]

df_plot = df.loc[pd.Timestamp('2020-09-01 00:00'):pd.Timestamp('2021-09-01 00:00'), columns_of_interest]

imputer = TimeSeriesImputer()
scaler = StandardScaler()
data = imputer.fit_transform(df_plot)
data = scaler.fit_transform(data)
df_plot = pd.DataFrame(data, columns=df_plot.columns, index=df_plot.index)

df_plot.plot()

# from tslearn.clustering import TimeSeriesKMeans
# model = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10)
# model.fit(data)

# values = dataset.values
# # specify columns to plot
# groups = [0, 1, 2, 3, 5, 6, 7]
# i = 1
# # plot each column
# pyplot.figure()
# for group in groups:
# 	pyplot.subplot(len(groups), 1, i)
# 	pyplot.plot(values[:, group])
# 	pyplot.title(dataset.columns[group], y=0.5, loc='right')
# 	i += 1
# pyplot.show()
