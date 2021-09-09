from typing import Tuple
import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import dump, load
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError

from preprocessing import ARIMAPreprocessor


class InnerConv1DBlock(tf.keras.Model):
    def __init__(self, filters: int, h: int, kernel_size: int, neg_slope: float = .01, dropout: float = .5,
                 name: str = ''):
        super(InnerConv1DBlock, self).__init__(name=name)
        self.conv1d = tf.keras.layers.Conv1D(h * filters, kernel_size, padding='same')
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
    def __init__(self, kernel_size: int, h: int):
        super(SciBlock, self).__init__()
        self.kernel_size = kernel_size
        self.h = h

        self.split = Split()
        self.exp = Exp()
        self.multiply = tf.keras.layers.Multiply()
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        _, _, filters = input_shape

        self.psi = InnerConv1DBlock(filters, self.h, self.kernel_size, name='psi')
        self.phi = InnerConv1DBlock(filters, self.h, self.kernel_size, name='phi')
        self.eta = InnerConv1DBlock(filters, self.h, self.kernel_size, name='eta')
        self.rho = InnerConv1DBlock(filters, self.h, self.kernel_size, name='rho')

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
    def __init__(self):
        super(Interleave, self).__init__()

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
        return self.interleave(inputs)


class SciNet(tf.keras.Model):
    def __init__(self, output_length: int, level: int, h: int, kernel_size: int):
        super(SciNet, self).__init__()
        self.level = level
        self.h = h
        self.kernel_size = kernel_size

        # self.sciblocks = [SciBlock(kernel_size, h) for i in range(2 ** self.level)]
        self.interleave = Interleave()
        self.add = tf.keras.layers.Add()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(output_length, kernel_regularizer=L1L2(0.0001, 0.01))

        assert level == 3
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

        x11, x12 = self.sciblock11(input_tensor)

        x21, x22 = self.sciblock21(x11)
        x23, x24 = self.sciblock22(x12)

        x31, x32 = self.sciblock31(x21)
        x33, x34 = self.sciblock32(x22)
        x35, x36 = self.sciblock33(x23)
        x37, x38 = self.sciblock34(x24)

        # x = self.interleave(outputs)
        x = self.interleave([x31, x32, x33, x34, x35, x36, x37, x38])
        x = self.add([x, input_tensor])

        x = self.flatten(x)
        x = self.dense(x)
        return x


# Parametres
degree_of_differencing = 1
look_back_window, forecast_horizon = 168, 3
batch_size = 64
learning_rate = 5e-3
h, kernel_size, level = 1, 5, 3
stride = look_back_window + forecast_horizon
data_filepath = 'ETH-USD-2020-06-01.csv'
y_col = 'close'

# Load and preprocess data
data = pd.read_csv(data_filepath)

train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)

# Make model
model = SciNet(forecast_horizon, level, h, kernel_size)
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # clipvalue=0.5 ?
model.compile(optimizer=opt, loss='mse')
print(model.summary)
# tf.keras.utils.plot_model(model, to_file='modelDiagram.png', show_shapes=True)

# Train model
preprocessor = ARIMAPreprocessor(y_col, look_back_window, forecast_horizon, stride, degree_of_differencing)
X_train, y_train = preprocessor.fit_transform(train_data)
print(f'Input shape: {X_train.shape}, {y_train.shape}')
early_stopping = EarlyStopping(monitor='val_loss', patience=100, min_delta=0, verbose=1, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.25, batch_size=batch_size, epochs=100000,
                    callbacks=[early_stopping])

# Generate new id, then save model, parser and relevant files
existing_ids = [int(name) for name in os.listdir('saved-models/') if name.isnumeric()]
run_id = random.choice(list(set(range(0, 1000)) - set(existing_ids)))
save_directory = f'saved-models/regressor/{run_id:03d}/'
os.makedirs(os.path.dirname(save_directory), exist_ok=True)

# Save model, preprocessor and training history
model.save(save_directory)
with open(save_directory + 'preprocessor', 'wb') as f:
    dump(preprocessor, f, compress=3)
pd.DataFrame(history.history).to_csv(save_directory + 'train_history.csv')

# Plot accuracy
# plt.plot(history.history['mean_absolute_error'])
# plt.plot(history.history['val_mean_absolute_error'])
# plt.title('model accuracy')
# plt.ylabel('mean absolute error')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper right')
# plt.savefig(save_directory + 'accuracy.png')
# plt.clf()

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig(save_directory + 'loss.png')

# Evaluate
# model = tf.saved_model.load(save_directory)
# with open(f'saved-models/regressor/111/preprocessor', 'rb') as f:
#     preprocessor = load(f)
X_test, y_test = preprocessor.transform(test_data)
scores = model.evaluate(X_test, y_test)

# Save evaluation results
if not isinstance(scores, list):
    scores = [scores]
row = [run_id] + scores + [pd.Timestamp.now(tz='Australia/Melbourne')]
try:
    df_scores = pd.read_csv('saved-models/scores.csv')
    df_scores.loc[len(df_scores)] = row
except FileNotFoundError:
    df_scores = pd.DataFrame([row], columns=['id'] + list(model.metrics_names) + ['time'])
df_scores.to_csv('saved-models/scores.csv', index=False)

# Predict
# model = load_model(f'saved-models/regressor/{run_id}/')
# with open(f'saved-models/regressor/{run_id:03d}/preprocessor', 'rb') as f:
#     preprocessor = load(f)
y_pred = model.predict(X_test)
comparison = np.vstack([y_pred.reshape(-1), y_test.reshape(-1)]).transpose()
df = pd.DataFrame(comparison, columns=['Prediction', 'Actual'])
df.to_csv('comparison.csv')
