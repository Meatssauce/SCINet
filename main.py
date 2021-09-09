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
from SCINet import *


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
