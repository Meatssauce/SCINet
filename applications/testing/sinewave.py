import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping

from SCINet import StackedSCINet, SCINet, StackedSCINetLoss, Identity
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gtda.time_series import SlidingWindow


def make_model(input_shape):
    # Model with a single SCINet
    # model = tf.keras.Sequential([
    #     tf.keras.Input(shape=(input_shape[1], input_shape[2]), name='inputs'),
    #     SCINet(horizon, features=input_shape[-1], levels=L, h=h, kernel_size=kernel_size)
    # ])

    # Model with StackedSCINEt
    inputs = tf.keras.Input(shape=(input_shape[1], input_shape[2]), name='lookback_window')
    x = StackedSCINet(horizon=horizon, features=input_shape[-1], stacks=K, levels=L, h=h,
                      kernel_size=kernel_size, regularizer=(l1, l2))(inputs)
    outputs = Identity(name='outputs')(x[-1])
    intermediates = Identity(name='intermediates')(x)
    model = tf.keras.Model(inputs=inputs, outputs=[outputs, intermediates])

    model.summary()
    tf.keras.utils.plot_model(model, to_file=os.path.join(output_dir, 'modelDiagram.png'), show_shapes=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss={
                      # 'outputs': 'mse',
                      'intermediates': StackedSCINetLoss()
                  },
                  metrics={'outputs': ['mse', 'mae']}
                  )

    return model


# Parameters
lag_length, horizon = 256, 48
batch_size = 16
learning_rate = 9e-3
h, kernel_size, L, K = 4, 5, 3, 2
l1, l2 = 0.001, 0.1

# Create output directory
output_dir = os.path.join('logs', 'sinewave', datetime.now().strftime('%HH%MM %d-%b-%Y'))
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

model = make_model(X_train.shape)
early_stopping = EarlyStopping(monitor='val_loss', patience=50, min_delta=0, verbose=1, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=500,
                    callbacks=[early_stopping])

# Save model and training history
model.save(output_dir)
pd.DataFrame(history.history).to_csv(os.path.join(output_dir, 'train_history.csv'))

# Plot some metrics
plt.plot(history.history['outputs_mae'])
plt.plot(history.history['val_outputs_mae'])
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
model = tf.keras.models.load_model(output_dir, custom_objects={'StackedSCINetLoss': StackedSCINetLoss()})

# Evaluate
scores = model.evaluate(X_test, y_test)
print(scores)

# Predict
y_pred, _ = model.predict(X_test)
print(np.mean(np.abs((y_pred - y_test))))  # manual mae for sanity check
