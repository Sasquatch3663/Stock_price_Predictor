import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Download Stock Data
ticker = "AAPL"  # Change to any stock symbol (TSLA, MSFT, etc.)
data = yf.download(ticker, start="2015-01-01", end="2023-01-01")

# Use only 'Close' price
closing_prices = data["Close"].values.reshape(-1, 1)

# 2. Normalize Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_prices)

# 3. Create Training Data (60 timesteps)
X_train, y_train = [], []
time_step = 60
for i in range(time_step, len(scaled_data)):
    X_train.append(scaled_data[i-time_step:i, 0])
    y_train.append(scaled_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# 4. Build LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")

# 5. Train the Model
model.fit(X_train, y_train, batch_size=32, epochs=50)

# 6. Predict Next 30 Days
last_60_days = scaled_data[-time_step:]
pred_input = last_60_days.reshape(1, time_step, 1)

future_predictions = []
for _ in range(30):
    pred = model.predict(pred_input)[0][0]
    future_predictions.append(pred)
    
    # Append prediction and remove first element to maintain 60 timesteps
    new_input = np.append(pred_input[0, 1:, 0], pred)
    pred_input = new_input.reshape(1, time_step, 1)

# Convert predictions back to original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# 7. Plot Results
plt.figure(figsize=(10,6))
plt.plot(closing_prices, color="blue", label="Historical Price")
plt.plot(range(len(closing_prices), len(closing_prices) + 30), future_predictions, color="red", label="Future Predictions (30 days)")
plt.title(f"{ticker} Stock Price Prediction (Next 30 Days)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
