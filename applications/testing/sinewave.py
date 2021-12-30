import os.path
import random

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping

from SCINet import StackedSCINet, SCINet, SCINetEndpoint
import tensorflow as tf
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


def make_model(input_shape, output_shape):
    # model = tf.keras.Sequential([
    #     tf.keras.Input(shape=(input_shape[1], input_shape[2]), name='inputs'),
    #     SciNet(horizon, features=input_shape[-1], levels=L, h=h, kernel_size=kernel_size)
    # ])

    inputs = tf.keras.Input(shape=(input_shape[1], input_shape[2]), name='inputs')
    targets = tf.keras.Input(shape=(output_shape[1], output_shape[2]), name='targets')
    predictions = StackedSCINet(horizon=horizon, features=input_shape[-1], stacks=K, levels=L, h=h,
                                kernel_size=kernel_size,
                                regularizer=(l1, l2))(inputs, targets)
    # logits = SciNet(horizon, features=input_shape[-1], levels=L, h=h, kernel_size=kernel_size, name='logits')(inputs)
    # predictions = SciNetEndpoint()(targets, logits)
    model = tf.keras.Model(inputs=[inputs, targets], outputs=predictions)

    model.summary()
    tf.keras.utils.plot_model(model, to_file=os.path.join(output_dir, 'modelDiagram.png'), show_shapes=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  # loss='mse',
                  # metrics=['mean_squared_error']
                  )
    return model


lag_length, horizon = 128, 24
batch_size = 16
learning_rate = 9e-3
h, kernel_size, L, K = 4, 5, 3, 2
l1, l2 = 0.001, 0.1

# Create output directory
output_dir = os.path.join('logs')
os.makedirs(output_dir, exist_ok=True)

# Generate data
time_steps = np.arange(0, 60, 0.01)
# col1 = [random.random() * 1000 for _ in range(50000)]
# col2 = [random.random() * 3000 for _ in range(50000)]
col1 = 4 * np.sin(time_steps)
col2 = 3 * np.sin(time_steps) + 3 * np.cos(2 * time_steps)
data = np.stack([col1, col2], axis=1)  # (?, 3)

# Plot data
plt.plot(col1)
plt.plot(col2)
plt.title('data')
plt.ylabel('y')
plt.xlabel('x')
plt.legend(['col1', 'col2'], loc='upper right')
plt.savefig(os.path.join(output_dir, 'data.png'))
plt.clf()

# split data -- train:val:test == 6:2:2
train_data, test_data = train_test_split(data, test_size=0.4, shuffle=False)
val_data, test_data = train_test_split(test_data, test_size=0.5, shuffle=False)

from gtda.time_series import SlidingWindow

windows = SlidingWindow(size=lag_length + horizon, stride=lag_length + horizon)
train_data = windows.fit_transform(train_data)

windows = SlidingWindow(size=lag_length + horizon, stride=lag_length + horizon)
val_data = windows.fit_transform(val_data)
test_data = windows.transform(test_data)

horizon = 48  # how many steps of y to predict
X_train, y_train = train_data[:, :-horizon, :], train_data[:, -horizon:, :]
X_val, y_val = val_data[:, :-horizon, :], val_data[:, -horizon:, :]
X_test, y_test = test_data[:, :-horizon, :], test_data[:, -horizon:, :]

model = make_model(X_train.shape, y_train.shape)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, min_delta=0, verbose=1, restore_best_weights=True)
history = model.fit(
    # X_train, y_train,
    # validation_data=(X_val, y_val),
    {'inputs': X_train, 'targets': y_train},
    validation_data={'inputs': X_val, 'targets': y_val},
    batch_size=batch_size, epochs=600, callbacks=[early_stopping])

# Save model, preprocessor and training history
model.save(output_dir)
pd.DataFrame(history.history).to_csv(os.path.join(output_dir, 'train_history.csv'))

# Plot accuracy
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model error')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig(os.path.join(output_dir, 'error.png'))
plt.clf()

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig(os.path.join(output_dir, 'error.png'))

# Evaluate
scores = model.evaluate(
    # X_test, y_test,
    {'inputs': X_test, 'targets': y_test}
)
print(scores)

# Predict
y_preds = model(
    # X_test,
    {'inputs': X_test, 'targets': y_test}
)
print(np.mean(np.abs((y_preds - y_test))))
