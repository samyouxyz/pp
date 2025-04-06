import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Simulate data
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", end="2025-04-06", freq="D")
prices = np.random.normal(loc=30000, scale=5000, size=len(dates))
data = pd.DataFrame({"Date": dates, "Close": prices})
data.set_index("Date", inplace=True)


# Prepare the data
def create_dataset(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

time_steps = 60
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - time_steps :]

X_train, y_train = create_dataset(train_data, time_steps)
X_test, y_test = create_dataset(test_data, time_steps)

# Reshape X for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform
train_predict = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Corrected plotting
train_dates = data.index[time_steps:train_size]
test_dates = data.index[train_size:]
test_dates = test_dates[-len(y_test_inv) :]

plt.figure(figsize=(12, 6))
plt.plot(train_dates, y_train_inv[:, 0], label="Actual Train Price")
plt.plot(train_dates, train_predict[:, 0], label="Predicted Train Price")
plt.plot(test_dates, y_test_inv[:, 0], label="Actual Test Price")
plt.plot(test_dates, test_predict[:, 0], label="Predicted Test Price")
plt.title("Bitcoin Price Prediction with LSTM")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# Calculate RMSE
train_rmse = np.sqrt(np.mean((y_train_inv[:, 0] - train_predict[:, 0]) ** 2))
test_rmse = np.sqrt(np.mean((y_test_inv[:, 0] - test_predict[:, 0]) ** 2))
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
