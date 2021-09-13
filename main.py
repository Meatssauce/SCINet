from typing import Tuple
import os
import random
import pandas as pd
import numpy as np
from joblib import dump, load
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError

from preprocessing import ARIMAPreprocessor
from SCINet import SciNet, StackedSciNet


# Make model
def make_model(input_shape, output_shape):
    inputs = tf.keras.Input(shape=(input_shape[1], input_shape[2]), name='inputs')
    # y = SciNet(forecast_horizon, levels=levels, h=h, kernel_size=kernel_size, regularizer=(0.001, 0.01))(x)
    # model = tf.keras.Model(x, y)
    targets = tf.keras.Input(shape=(output_shape[1]), name='targets')
    predictions = StackedSciNet(horizon, stacks=K, levels=L, h=h, kernel_size=kernel_size,
                                regularizer=(l1, l2))(inputs, targets)
    model = tf.keras.Model(inputs=[inputs, targets], outputs=predictions)

    model.summary()
    tf.keras.utils.plot_model(model, to_file='modelDiagram.png', show_shapes=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse',
                  metrics=['mean_squared_error', 'mean_absolute_error'])
    return model


# Parametres
data_filepath = 'crypto_data/Crypto-USD-2019-09-01-00-00.csv'
y_col = 'ETH/close'
index_col = 'time'
# data_filepath = 'ETDataset-main/ETT-small/ETTh1.csv'
# y_col = 'OT'
# index_col = 'date'
degree_of_differencing = 0
look_back_window, horizon = 168, 3
batch_size = 64
learning_rate = 9e-3
h, kernel_size, L, K = 4, 5, 3, 1
l1, l2 = 0.001, 0.1
split_strides = look_back_window + horizon  # unsure if any value lower than this would cause data leak
# split_strides = 1


if __name__ == '__main__':
    # Load and preprocess data
    data = pd.read_csv(data_filepath, index_col=index_col)

    train_data = data[:int(0.6 * len(data))]
    val_data = data[int(0.6 * len(data)):int(0.8 * len(data))]
    test_data = data[int(0.8 * len(data)):]

    # Train model
    preprocessor = ARIMAPreprocessor(y_col, look_back_window, horizon, split_strides, degree_of_differencing,
                                     relative_diff=False, scaling='standard')
    X_train, y_train = preprocessor.fit_transform(train_data)
    X_val, y_val = preprocessor.transform(val_data)
    print(f'Input shape: X{X_train.shape}, y{y_train.shape}')

    model = make_model(X_train.shape, y_train.shape)
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, min_delta=0, verbose=1, restore_best_weights=True)
    history = model.fit({'inputs': X_train, 'targets': y_train},
                        validation_data={'inputs': X_val, 'targets': y_val},
                        batch_size=batch_size, epochs=800, callbacks=[early_stopping])

    # Generate new id and create save directory
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
    # run_id = 186
    # model = load_model(f'saved-models/regressor/{run_id:03d}/')
    # with open(f'saved-models/regressor/{run_id:03d}/preprocessor', 'rb') as f:
    #     preprocessor = load(f)
    X_test, y_test = preprocessor.transform(test_data)
    scores = model.evaluate({'inputs': X_test, 'targets': y_test})

    # Save evaluation results
    if not isinstance(scores, list):
        scores = [scores]
    row = [run_id] + scores + [pd.Timestamp.now(tz='Australia/Melbourne')]
    try:
        df_scores = pd.read_csv('saved-models/scores.csv')
        df_scores.loc[len(df_scores)] = row
    except (FileNotFoundError, ValueError):
        df_scores = pd.DataFrame([row], columns=['id'] + list(model.metrics_names) + ['time'])
    df_scores.to_csv('saved-models/scores.csv', index=False)

    # Predict
    # y_test is only used to calculate loss, how to get rid of it?
    y_pred = model.predict({'inputs': X_test, 'targets': y_test})
    y_pred = preprocessor.y_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test = preprocessor.y_scaler.inverse_transform(y_test.reshape(-1, 1))
    comparison = np.hstack([y_pred, y_test])
    df = pd.DataFrame(comparison, columns=['Predicted', 'Actual'])
    df.to_csv(f'saved-models/regressor/{run_id:03d}/comparison.csv')
