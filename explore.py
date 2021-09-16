import pandas as pd
import numpy as np
import re
from joblib import dump
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

from preprocessing import TimeSeriesImputer, CryptoPreprocessor


def lower_granularity(df, interval):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise IndexError('Dataframe must have datetime index')

    df = df.sort_index()
    prev_interval = df.index[1] - df.index[0]
    if np.abs(interval) < np.abs(prev_interval):
        raise ValueError('Cannot increase granularity')
    if (bin_width := interval / prev_interval) % 1 != 0:
        raise ValueError('New interval must evenly divide the current interval')
    bin_width = int(bin_width)

    # Resample data from most recent to least recent
    df2 = df.reset_index(drop=True)
    data = {'low': np.vstack([df2.loc[i - bin_width:i, 'low'].min().values
                              for i in range(len(df2) - 1, -1, -bin_width)]),
            'high': np.vstack([df2.loc[i - bin_width:i, 'high'].max().values
                               for i in range(len(df2) - 1, -1, -bin_width)]),
            'open': np.vstack([df2.loc[i - bin_width if i >= bin_width else 0, 'open'].values
                               for i in range(len(df2) - 1, -1, -bin_width)]),
            'close': np.vstack([df2.loc[i, 'close'].values
                                for i in range(len(df2) - 1, -1, -bin_width)]),
            'price': np.vstack([df2.loc[i - bin_width:i, 'price'].mean().values
                                for i in range(len(df2) - 1, -1, -bin_width)]),
            'volume': np.vstack([df2.loc[i - bin_width:i, 'volume'].mean().values
                                 for i in range(len(df2) - 1, -1, -bin_width)]),
            }
    df2 = df[::-1][::bin_width]
    for category, values in data.items():
        df2.loc[:, category] = values

    return df2[::-1]


# Exploring cryptocurrency data

# # Load and index data
# df = pd.read_csv('crypto_data/USD-2017-01-01.csv', index_col='time')
# df.index = pd.to_datetime(df.index)
#
# first_valid_idx = max([df[col].first_valid_index() for col in df.columns])
# last_valid_idx = min([df[col].last_valid_index() for col in df.columns])
# df = df.loc[first_valid_idx:last_valid_idx, :]
#
# # Make new multi-index dataframe for average price
# df_price = (df.loc[:, 'open'] + df.loc[:, 'close']) / 2
#
# # Add concatenate two dataframes
# df = pd.concat([df, df_price], axis=1).sort_values(by=['category', 'coin'], axis=1)

# Load and index data
df = pd.read_csv('crypto_data/USD-2021-06-17-2021-09-12.csv', index_col=0, header=[0, 1])
df.index = pd.to_datetime(df.index)

preprocessor = CryptoPreprocessor(splitXy=False, target_coin='ETH')
df = preprocessor.fit_transform(df)
df = df.loc[:, ['low', 'price', 'high']]
plt.plot(df)
plt.savefig('test.png')

