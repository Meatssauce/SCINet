import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from main import degree_of_differencing, look_back_window, forecast_horizon, stride, data_filepath, y_col
from preprocessing import ARIMAPreprocessor


# Load and preprocess data
data = pd.read_csv(data_filepath)
columns = data.drop(columns=['time']).columns
time = data['time']

# train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)
train_data = data[:int(0.6 * len(data))]
val_data = data[int(0.6 * len(data)):int(0.8 * len(data))]
test_data = data[int(0.8 * len(data)):]

degree_of_differencing = 2
preprocessor = ARIMAPreprocessor(y_col, look_back_window, forecast_horizon, stride, degree_of_differencing,
                                 splitXy=False)
data = preprocessor.fit_transform(train_data)

start = 0 if degree_of_differencing < 1 else 1

new_columns = [col + f'{i}' for i in range(start, degree_of_differencing + 1) for col in columns]
data = pd.DataFrame(data, columns=new_columns)
data['time'] = pd.to_datetime(time)
data.plot('time', 'close0')

print()
