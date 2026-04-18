import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# --- CONFIGURATION ---
START_DATE = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
TICKER = "^NSEI" # Yahoo Finance ticker for Nifty 50

st.set_page_config(page_title="Nifty 50 Predictor", layout="wide")

# --- UI HEADER ---
st.title("📈 Nifty 50 Future Price Predictor")

# --- IMPORTANT ALERTS & DISCLAIMERS ---
st.warning("""
**🚨 IMPORTANT ALERT:** The stock market is highly sensitive to real-world events. The predictions shown here **do not** and **cannot** account for sudden breaking news, global events, economic shifts, or new rules and policies implemented by the government. Such events can drastically and immediately affect stock prices.
""")

st.markdown("""
*Disclaimer: This app uses historical data and time-series forecasting (Prophet) to project past trends into the future. These predictions should **not** be used for financial trading or investment decisions.*
""")

# --- DATA FETCHING ---
@st.cache_data
def load_data(ticker):
    # Using Ticker().history() is often more stable than yf.download()
    stock = yf.Ticker(ticker)
    data = stock.history(start=START_DATE, end=TODAY)
    data.reset_index(inplace=True)
    
    # Prophet requires timezone-naive dates. This removes the timezone info.
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
        
    return data

data_load_state = st.text('Loading real-time Nifty 50 data...')
data = load_data(TICKER)
data_load_state.text('Loading real-time Nifty 50 data... Done!')

# --- RAW DATA VIEW ---
st.subheader("Raw Historical Data")
st.write(data.tail())

# --- PLOT HISTORICAL DATA ---
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close Price"))
    fig.layout.update(title_text='Nifty 50 Historical Close Price', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

# --- FORECASTING MODEL ---
st.subheader("🔮 Forecast Future Prices")

# Slider to choose how many years into the future to predict
n_years = st.slider('Select years of prediction:', 1, 5)
period = n_years * 365

# Prepare data for Prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Drop any rows that have missing values (NaNs)
df_train = df_train.dropna()

# Safety Check: If Yahoo Finance returned empty data, stop the app gracefully
if len(df_train) < 2:
    st.error("🚨 **Data Error:** Not enough data was fetched from Yahoo Finance to make a prediction. This usually happens if Yahoo Finance is temporarily blocking cloud servers. Please try again in a few minutes.")
    st.stop() # This halts the script here so it doesn't crash the app

# Initialize and fit the model
m = Prophet(daily_seasonality=False)
m.fit(df_train)

# Create future dataframe and predict
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# --- DISPLAY FORECAST ---
st.subheader(f"Forecast Data ({n_years} Years)")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

st.write("### Forecast Plot")
fig1 = plot_plotly(m, forecast)
fig1.layout.update(title_text='Nifty 50 Price Forecast', xaxis_rangeslider_visible=True)
st.plotly_chart(fig1, use_container_width=True)

st.write("### Forecast Components")
st.write("This shows the underlying trends and yearly/weekly seasonality found by the model.")
fig2 = m.plot_components(forecast)
st.write(fig2)
