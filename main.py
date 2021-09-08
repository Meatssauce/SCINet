from typing import Tuple
import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import dump, load
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError


class InnerConv1DBlock(tf.keras.Model):
    def __init__(self, filters, h, kernel_size, neg_slope=.01, dropout=.5):
        super(InnerConv1DBlock, self).__init__()
        self.conv1d = tf.keras.layers.Conv1D(h * filters, kernel_size, padding='same', dtype='float32')
        self.leakyrelu = tf.keras.layers.LeakyReLU(neg_slope)

        self.dropout = tf.keras.layers.Dropout(dropout)

        self.conv1d2 = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')
        self.tanh = tf.keras.activations.tanh

    def call(self, input_tensor):
        x = self.conv1d(input_tensor)
        x = self.leakyrelu(x)

        x = self.dropout(x)

        x = self.conv1d2(x)
        x = self.tanh(x)
        return x


class Exp(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Exp, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.exp(inputs)


class Split(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Split, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs[:, ::2], inputs[:, 1::2]


class SciBlock(tf.keras.Model):
    def __init__(self, kernel_size, h):
        super(SciBlock, self).__init__()
        self.kernel_size = kernel_size
        self.h = h

        self.split = Split()
        self.exp = Exp()
        self.multiply = tf.keras.layers.Multiply()
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        _, _, filters = input_shape

        self.psi = InnerConv1DBlock(filters, self.h, self.kernel_size)
        self.phi = InnerConv1DBlock(filters, self.h, self.kernel_size)
        self.eta = InnerConv1DBlock(filters, self.h, self.kernel_size)
        self.rho = InnerConv1DBlock(filters, self.h, self.kernel_size)

    def call(self, input_tensor):
        F_odd, F_even = self.split(input_tensor)

        F_s_odd = self.multiply([F_odd, self.exp(self.phi(F_even))])
        F_s_even = self.multiply([F_even, self.exp(self.psi(F_s_odd))])

        F_prime_odd = self.add([F_s_odd, self.rho(F_s_even)])
        F_prime_even = self.add([F_s_even, self.eta(F_s_odd)])

        # F_s_odd = F_odd * self.exp(self.phi(F_even))
        # F_s_even = F_even * self.exp(self.psi(F_s_odd))
        #
        # F_prime_odd = F_s_odd + self.rho(F_s_even)
        # F_prime_even = F_s_even + self.eta(F_s_odd)
        return F_prime_odd, F_prime_even


class Interleave(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Interleave, self).__init__(**kwargs)

    def interleave(self, slices):
        if not slices:
            return slices
        elif len(slices) == 1:
            return slices[0]

        mid = len(slices) // 2

        even = self.interleave(slices[:mid])
        odd = self.interleave(slices[mid:])

        shape = tf.shape(even)
        return tf.reshape(tf.stack([even, odd], axis=3), (shape[0], shape[1]*2, shape[2]))

    def call(self, inputs):
        # print(inputs)
        return self.interleave(inputs)


class SciNet(tf.keras.Model):
    def __init__(self, output_length: Tuple[int, int], level: int, h: int, kernel_size: int):
        super(SciNet, self).__init__()
        self.level = level
        self.h = h
        self.kernel_size = kernel_size

        # self.sciblocks = [SciBlock(kernel_size, h) for i in range(2 ** self.level)]
        self.interleave = Interleave()
        self.add = tf.keras.layers.Add()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(output_length)

        self.sciblock11 = SciBlock(kernel_size, h)
        self.sciblock21 = SciBlock(kernel_size, h)
        self.sciblock22 = SciBlock(kernel_size, h)
        self.sciblock31 = SciBlock(kernel_size, h)
        self.sciblock32 = SciBlock(kernel_size, h)
        self.sciblock33 = SciBlock(kernel_size, h)
        self.sciblock34 = SciBlock(kernel_size, h)

    def call(self, input_tensor):
        # cascade input down a binary tree of sci-blocks
        # inputs, outputs = [input_tensor], []
        # i = 0
        # while i < 2 ** self.level - 1:  # i < max_blocks
        #     outputs = [element for tensor in inputs for element in self.sciblocks[i](tensor)]
        #     i = len(outputs)
        #     inputs = outputs
        # print(input_tensor)
        x11, x12 = self.sciblock11(input_tensor)

        x21, x22 = self.sciblock21(x11)
        x23, x24 = self.sciblock22(x12)

        x31, x32 = self.sciblock31(x21)
        x33, x34 = self.sciblock32(x22)
        x35, x36 = self.sciblock33(x23)
        x37, x38 = self.sciblock34(x24)
        # print(f'x31{x31.shape}\nx32{x32.shape}\nx33{x33.shape}\nx34{x34.shape}\nx35{x35.shape}\nx36{x36.shape}\n'
        #       f'x37{x37.shape}\nx38{x38.shape}')
        x = self.interleave([x31, x32, x33, x34, x35, x36, x37, x38])
        # x = self.interleave(outputs)
        x = self.add([x, input_tensor])
        # print(x)
        x = self.flatten(x)
        # print(x)
        x = self.dense(x)
        # print(x)
        return x


# split a sequence into X, y samples
def split_sequence(sequence, look_back_window, forecast_horizon, stride=1):
    X, y = [], []
    for i in range(0, len(sequence), stride):
        # find the end x and y
        end_ix = i + look_back_window
        end_iy = end_ix + forecast_horizon

        # check if there is enough elements to fill this x, y pair
        if end_iy > len(sequence):
            break

        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_iy - 1 if forecast_horizon == 1 else end_ix:end_iy]
        X.append(seq_x)
        y.append(seq_y)

    return np.asarray(X), np.asarray(y)


# Parametres
look_back_window, forecast_horizon = 168, 3
h, kernel_size, level = 1, 5, 3
data_filepath = 'ETH-USD-2020-06-01.csv'

# if os.path.isfile('__cache'):
#     with open('__cache', 'rb') as f:
#         X, y = load(f)
# else:
#     data = pd.read_csv(data_filepath)
#     data = (data - data.mean()) / data.var()  # z-score transformation
#     X, y = split_sequence(data.values, look_back_window, forecast_horizon)
#     y_idx = data.columns.tolist().index('close')
#     y = y[:, :, y_idx]
#     del data
#     with open('__cache', 'wb') as f:
#         dump((X, y), f)
# print(f'X{X.shape}\ny{y.shape}')

# Load and preprocess data
data = pd.read_csv(data_filepath)
y_idx = data.columns.tolist().index('close')

train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)

train_data = (train_data - train_data.mean()) / (train_data.var() ** (1/2))  # z-score transformation
test_data = (test_data - train_data.mean()) / (train_data.var() ** (1/2))

X_train, y_train = split_sequence(train_data.values, look_back_window, forecast_horizon,
                                  look_back_window + forecast_horizon)
y_train = y_train[:, :, y_idx]

X_test, y_test = split_sequence(test_data.values, look_back_window, forecast_horizon,
                                look_back_window + forecast_horizon)
y_test = y_test[:, :, y_idx]

# Make model
model = SciNet(forecast_horizon, level, h, kernel_size)
# optimizer = optimizers.Adam(5e-3, clipvalue=0.5)
model.compile(optimizer='adam', loss='mse')
print(model.summary)
# tf.keras.utils.plot_model(model, to_file='modelDiagram.png', show_shapes=True)

# Train model
# Distributions = namedtuple('Distributions', 'means variances')
# # preprocessor = StandardScaler()
# # X_train = preprocessor.fit_transform(X_train)
# X_train = X_train.reshape(-1, X_train.shape[-1])
# X_distro = Distributions(X_train.mean(axis=0), X_train.var(axis=0))
# X_train = (X_train - train_distro.means) / np.sqrt(train_distro.variances)
# X_train = X_train.reshape(-1, look_back_window, X_train.shape[-1])
#
# y_distro = Distributions(y_train.mean(axis=0), y_train.var(axis=0))
# y_train = (y_train - y_distro.means) / np.sqrt(y_distro.variances)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=0, verbose=1, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.25, batch_size=4, epochs=1000, callbacks=[early_stopping])

# Generate new id, then save model, parser and relevant files
existing_ids = [int(name) for name in os.listdir('saved-models/') if name.isnumeric()]
run_id = random.choice(list(set(range(0, 1000)) - set(existing_ids)))
save_directory = f'saved-models/{run_id:03d}/'
os.makedirs(os.path.dirname(save_directory), exist_ok=True)

model.save(save_directory + 'SCINET_regressor.hdf5')
# with open(save_directory + 'train_data_distributions', 'wb') as f:
#     dump((X_distro, y_distro), f, compress=3)
pd.DataFrame(history.history).to_csv(save_directory + 'train_history.csv')

# Plot accuracy
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model accuracy')
plt.ylabel('mean absolute error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig(save_directory + 'accuracy.png')
plt.clf()

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig(save_directory + 'loss.png')

# Evaluate
# run_id = 523
# model = load_model(f'saved-models/{run_id}/SCINET_regressor.hdf5')
# with open(f'saved-models/{run_id}/train_data_distributions', 'rb') as f:
#     X_distro, y_distro = load(f)
# X_test = preprocessor.transform(X_test)
# X_test = X_test.reshape(-1, X_test.shape[-1])
# X_test = (X_test - X_distro.means) / np.sqrt(X_distro.variances)
# X_test = X_test.reshape(-1, look_back_window, X_test.shape[-1])
# y_train = (y_train - y_distro.means) / np.sqrt(y_distro.variances)
scores = model.evaluate(X_test, y_test)
try:
    df_scores = pd.read_csv('saved-models/scores.csv')
    df_scores.loc[len(df_scores)] = [run_id] + scores + [pd.Timestamp.now(tz='Australia/Melbourne')]
except FileNotFoundError:
    row = [[run_id] + scores + [pd.Timestamp.now(tz='Australia/Melbourne')]]
    df_scores = pd.DataFrame(row, columns=['id'] + list(model.metrics_names) + ['time'])
df_scores.to_csv('saved-models/scores.csv', index=False)
