import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from SCINet import make_simple_scinet, make_simple_stacked_scinet, StackedSCINetLoss
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gtda.time_series import SlidingWindow


# Model hyperparameters
lag_length, horizon = 256, 48
learning_rate = 9e-3
h, kernel_size, L, K = 4, 5, 3, 2
kernel_regularizer = tf.keras.regularizers.L1L2(0.001, 0.01)

# Create output directory
output_dir = os.path.join('saved_models_with_logs', 'sinewave', datetime.now().strftime('%HH%MM %d-%b-%Y'))
os.makedirs(output_dir, exist_ok=True)

# Generate dummy data
time_steps = np.arange(0, 60, 0.01)
col1 = 4 * np.sin(time_steps)
col2 = 3 * np.sin(time_steps) + 3 * np.cos(2 * time_steps)
data = np.stack([col1, col2], axis=1)

# Plot data
plt.plot(col1)
plt.plot(col2)
plt.title('data')
plt.ylabel('y')
plt.xlabel('x')
plt.legend(['col1', 'col2'], loc='upper right')
plt.savefig(os.path.join(output_dir, 'data.png'))
plt.clf()

# Split data -- train:val:test == 6:2:2
train_data, test_data = train_test_split(data, test_size=0.4, shuffle=False)
val_data, test_data = train_test_split(test_data, test_size=0.5, shuffle=False)

# Segment into train/val/test examples (may use stride < size for train_data only but may cause data leak)
windows = SlidingWindow(size=lag_length + horizon, stride=lag_length + horizon)
train_data = windows.fit_transform(train_data)
val_data = windows.transform(val_data)
test_data = windows.transform(test_data)

# Split all time series segments into x and y
X_train, y_train = train_data[:, :-horizon, :], train_data[:, -horizon:, :]
X_val, y_val = val_data[:, :-horizon, :], val_data[:, -horizon:, :]
X_test, y_test = test_data[:, :-horizon, :], test_data[:, -horizon:, :]

diagram_path = os.path.join(output_dir, 'model_diagram.png')

# Proceed with SCINet
model = make_simple_scinet(X_train.shape, horizon=horizon, L=L, h=h, kernel_size=kernel_size,
                           learning_rate=learning_rate, kernel_regularizer=kernel_regularizer,
                           diagram_path=diagram_path)
early_stopping = EarlyStopping(monitor='val_loss', patience=50, min_delta=0, verbose=1, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=16, epochs=500,
                    callbacks=[early_stopping])

# Save model and training history
model.save(output_dir)
pd.DataFrame(history.history).to_csv(os.path.join(output_dir, 'train_history.csv'))

# Plot some metrics
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model error')
plt.ylabel('mean absolute error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig(os.path.join(output_dir, 'outputs_mae.png'))
plt.clf()

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig(os.path.join(output_dir, 'loss.png'))
plt.clf()

# Reconstruct/load model
model = tf.keras.models.load_model(output_dir)

# Evaluate
scores = model.evaluate(X_test, y_test)
print(scores)

# Predict
y_pred = model.predict(X_test)
print(np.mean(np.abs((y_pred - y_test))))  # manual mae for sanity check

# # Proceed with StackedSCINet
# model = make_simple_stacked_scinet(X_train.shape, horizon=horizon, K=K, L=L, h=h, kernel_size=kernel_size,
#                                    learning_rate=learning_rate, kernel_regularizer=kernel_regularizer,
#                                    diagram_path=diagram_path)
# early_stopping = EarlyStopping(monitor='val_loss', patience=50, min_delta=0, verbose=1, restore_best_weights=True)
# history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=16, epochs=500,
#                     callbacks=[early_stopping])
#
# # Save model and training history
# model.save(output_dir)
# pd.DataFrame(history.history).to_csv(os.path.join(output_dir, 'train_history.csv'))
#
# # Plot some metrics
# plt.plot(history.history['outputs_mae'])
# plt.plot(history.history['val_outputs_mae'])
# plt.title('model error')
# plt.ylabel('mean absolute error')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper right')
# plt.savefig(os.path.join(output_dir, 'outputs_mae.png'))
# plt.clf()
#
# # Plot loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper right')
# plt.savefig(os.path.join(output_dir, 'loss.png'))
# plt.clf()
#
# # Reconstruct/load model
# model = tf.keras.models.load_model(output_dir, custom_objects={'StackedSCINetLoss': StackedSCINetLoss()})
#
# # Evaluate
# scores = model.evaluate(X_test, y_test)
# print(scores)
#
# # Predict
# y_pred, _ = model.predict(X_test)
# print(np.mean(np.abs((y_pred - y_test))))  # manual mae for sanity check
