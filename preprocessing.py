import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def split_sequence(sequence, look_back_window: int, forecast_horizon: int, stride: int = 1):
    X, y = [], []
    for i in range(0, len(sequence), stride):
        # find the end x and y
        end_ix = i + look_back_window
        end_iy = end_ix + forecast_horizon

        # check if there is enough elements to fill this x, y pair
        if end_iy > len(sequence):
            break

        X.append(sequence[i:end_ix])
        y.append(sequence[end_iy - 1 if forecast_horizon == 1 else end_ix:end_iy])
    return np.asarray(X), np.asarray(y)


class TimeSeriesImputer(TransformerMixin):
    def __init__(self, method: str = 'linear', fail_save: TransformerMixin = SimpleImputer()):
        self.method = method
        self.fail_save = fail_save

    def fit(self, data):
        if self.fail_save:
            self.fail_save.fit(data)
        return self

    def transform(self, data):
        # Interpolate missing values in columns
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        data = data.interpolate(method=self.method, limit_direction='both')
        # spline or time may be better?

        if self.fail_save:
            data = self.fail_save.transform(data)

        return data


def difference(dataset, interval=1, relative=False, min_price=1e-04):
    delta = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        if relative:
            prev_price = dataset[i - interval]
            prev_price[prev_price == 0] = min_price
            value /= prev_price
        delta.append(value)
    return np.asarray(delta)


class TimeSeriesPreprocessor(TransformerMixin):
    def __init__(self, look_back_window: int, forecast_horizon: int, stride: int, diff_order: int,
                 relative_diff: bool = True, splitXy: bool = True, scaling: str = 'minmax'):
        if not (look_back_window > 0 and forecast_horizon > 0 and stride > 0):
            raise ValueError('look_back_window, forecast_horizon and stride must be positive')
        # if stride < look_back_window + forecast_horizon:
        #     raise PendingDeprecationWarning('Setting stride to less than look_back_window + forecast_horizon may not'
        #                                     'be supported in the future due to potential data leak.')
        super().__init__()
        self.look_back_window = look_back_window
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.diff_order = diff_order
        self.relative_diff = relative_diff
        self.splitXy = splitXy
        self.interpolation_imputer = TimeSeriesImputer(method='linear')

        if scaling == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling == 'standard':
            self.scaler = StandardScaler()
        elif scaling == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError('Invalid value for scaling')

    def fit_transform(self, data, **fit_params):
        # Fill missing values via interpolation
        data = self.interpolation_imputer.fit_transform(data)

        # Differencing
        diff = np.array(data)
        for d in range(1, self.diff_order + 1):
            diff = difference(diff, relative=self.relative_diff)
            data = np.append(data, np.pad(diff, pad_width=((d, 0), (0, 0))), axis=1)
        if self.diff_order > 0:
            data = data[:, diff.shape[1]:]

        # Scale
        # if self.diff_order < 1:
        data = self.scaler.fit_transform(data)

        if not self.splitXy:
            return data

        # Extract X, y from time series
        X, y = split_sequence(data, self.look_back_window, self.forecast_horizon, self.stride)

        return X, y

    def transform(self, data):
        # Fill missing values via interpolation
        data = self.interpolation_imputer.transform(data)

        # Differencing
        diff = np.array(data)
        for d in range(1, self.diff_order + 1):
            diff = difference(diff, relative=self.relative_diff)
            data = np.append(data, np.pad(diff, pad_width=((d, 0), (0, 0))), axis=1)
        if self.diff_order > 0:
            data = data[:, diff.shape[1]:]

        # Scale
        # if self.diff_order < 1:
        data = self.scaler.transform(data)

        if not self.splitXy:
            return data

        # Extract X, y
        X, y = split_sequence(data, self.look_back_window, self.forecast_horizon, self.stride)

        return X, y


class CryptoPreprocessor(TransformerMixin):
    def __init__(self, look_back_window: int = 168, forecast_horizon: int = 24, stride: int = 1, diff_order: int = 0,
                 relative_diff: bool = True, splitXy: bool = True, target_coin: str = None):
        if not (look_back_window > 0 and forecast_horizon > 0 and stride > 0):
            raise ValueError('look_back_window, forecast_horizon and stride must be positive')
        # if stride < look_back_window + forecast_horizon:
        #     raise PendingDeprecationWarning('Setting stride to less than look_back_window + forecast_horizon may not'
        #                                     'be supported in the future due to potential data leak.')
        super().__init__()
        self.look_back_window = look_back_window
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.diff_order = diff_order
        self.relative_diff = relative_diff
        self.splitXy = splitXy
        self.target_coin = target_coin
        self.interpolation_imputer = TimeSeriesImputer(method='linear')
        self.scaler = {}

    def fit_transform(self, df, **fit_params):
        # Fill missing values via interpolation
        df = pd.DataFrame(self.interpolation_imputer.fit_transform(df), index=df.index, columns=df.columns)

        # # Differencing
        # diff = np.array(data)
        # for d in range(1, self.diff_order + 1):
        #     diff = difference(diff, relative=self.relative_diff)
        #     data = np.append(data, np.pad(diff, pad_width=((d, 0), (0, 0))), axis=1)
        # if self.diff_order > 0:
        #     data = data[:, diff.shape[1]:]

        # Scale
        # if self.diff_order < 1:
        # todo: for each coin scale all categories by fit on price
        # self.scaler = {coin: StandardScaler() for coin in df.columns.get_level_values('coin')}
        # for coin in self.scaler.keys():
        #     df.loc[:, ('price', coin)] = self.scaler[coin].fit_transform(df)
        #     for
        # df = pd.DataFrame(self.scaler.fit_transform(df), index=df.index, columns=df.columns)

        # Remove market trend from
        df = df.sort_values(by=['category'], axis=1)
        other_coins = [col for col in df.columns.levels[1].unique() if col != 'ETH']
        df_mean = pd.DataFrame.copy(df.loc[:, (df.columns.levels[0], self.target_coin)])
        for cat in df.columns.levels[0].unique():
            df_mean.loc[:, (cat, self.target_coin)] = df.loc[:, (cat, other_coins)].mean(axis=1)
        df = df.loc[:, (df.columns.levels[0], self.target_coin)] - df_mean

        if not self.splitXy:
            return df

        # Extract X, y from time series
        X, y = split_sequence(df, self.look_back_window, self.forecast_horizon, self.stride)

        return X, y

    def transform(self, df):
        # Fill missing values via interpolation
        df = pd.DataFrame(self.interpolation_imputer.transform(df), index=df.index, columns=df.columns)

        # # Differencing
        # diff = np.array(data)
        # for d in range(1, self.diff_order + 1):
        #     diff = difference(diff, relative=self.relative_diff)
        #     data = np.append(data, np.pad(diff, pad_width=((d, 0), (0, 0))), axis=1)
        # if self.diff_order > 0:
        #     data = data[:, diff.shape[1]:]

        # Scale
        # if self.diff_order < 1:
        df = pd.DataFrame(self.scaler.transform(df), index=df.index, columns=df.columns)

        # Remove market trend from
        df = df.sort_values(by=['category'], axis=1)
        other_coins = [col for col in df.columns.levels[1].unique() if col != 'ETH']
        df_mean = pd.DataFrame.copy(df.loc[:, (df.columns.levels[0], self.target_coin)])
        for cat in df.columns.levels[0].unique():
            df_mean.loc[:, (cat, self.target_coin)] = df.loc[:, (cat, other_coins)].mean(axis=1)
        df = df.loc[:, (df.columns.levels[0], self.target_coin)] - df_mean

        if not self.splitXy:
            return df

        # Extract X, y
        X, y = split_sequence(df, self.look_back_window, self.forecast_horizon, self.stride)

        return X, y
