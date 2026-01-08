import streamlit as st
import requests
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Live Crypto Dashboard", layout="wide")
st.title("🚀 LIVE CRYPTOCURRENCY MARKET DASHBOARD")

# -------------------
# Fetch Live Data
# -------------------
def get_crypto_data(coin='bitcoin', days=30):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
    response = requests.get(url).json()
    prices = response['prices']  # [timestamp, price]
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Sidebar for coin selection
coins = ['bitcoin', 'ethereum', 'cardano', 'dogecoin', 'solana']
selected_coin = st.sidebar.selectbox("Select Cryptocurrency", coins)

# -------------------
# Show Price Chart
# -------------------
st.subheader(f"{selected_coin.capitalize()} Price Chart")
data = get_crypto_data(selected_coin, days=90)
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['timestamp'], y=data['price'], mode='lines', name='Price'))
fig.update_layout(xaxis_title='Date', yaxis_title='Price (USD)')
st.plotly_chart(fig, use_container_width=True)

# -------------------
# Simple Price Prediction (LSTM)
# -------------------
st.subheader("📈 Simple Price Prediction (Next 7 Days)")

# Prepare data
dataset = data['price'].values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create sequences
def create_sequences(data, time_step=10):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_step = 10
X, y = create_sequences(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X, y, epochs=5, batch_size=16, verbose=0)

# Predict next 7 days
temp_input = scaled_data[-time_step:].reshape(1, time_step, 1)
predictions = []
for _ in range(7):
    pred = model.predict(temp_input, verbose=0)
    predictions.append(pred[0,0])
    temp_input = np.append(temp_input[:,1:,:], [[pred]], axis=1)

pred_prices = scaler.inverse_transform(np.array(predictions).reshape(-1,1))

# Show Prediction
future_dates = pd.date_range(start=data['timestamp'].iloc[-1]+pd.Timedelta(days=1), periods=7)
pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': pred_prices.flatten()})
st.table(pred_df)

# Plot prediction
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=data['timestamp'], y=data['price'], mode='lines', name='Historical Price'))
fig2.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted Price'], mode='lines+markers', name='Predicted Price'))
fig2.update_layout(xaxis_title='Date', yaxis_title='Price (USD)')
st.plotly_chart(fig2, use_container_width=True)
