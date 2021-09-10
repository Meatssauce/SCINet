import pandas as pd
import matplotlib.pyplot as plt

from main import degree_of_differencing, look_back_window, forecast_horizon, stride, data_filepath, y_col, index_col
from preprocessing import ARIMAPreprocessor


# Load and preprocess data
data = pd.read_csv(data_filepath, index_col=index_col)
columns = data.columns

# train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)
train_data = data[:int(0.6 * len(data))]
val_data = data[int(0.6 * len(data)):int(0.8 * len(data))]
test_data = data[int(0.8 * len(data)):]

idx = train_data.index
preprocessor = ARIMAPreprocessor(y_col, look_back_window, forecast_horizon, stride, degree_of_differencing,
                                 splitXy=False, relative_diff=False, scaling='standard')
train_data = preprocessor.fit_transform(train_data)

start = 0 if degree_of_differencing < 1 else 1
new_columns = [col + f'{i}' for i in range(start, degree_of_differencing + 1) for col in columns]
train_data = pd.DataFrame(train_data, columns=new_columns)
train_data[index_col] = idx
train_data.plot(index_col, y_col+'0')

print()
