import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import io

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
with st.expander("Show Raw Historical Data"):
    st.write(data.tail())

# --- PLOT HISTORICAL DATA ---
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close Price"))
    fig.layout.update(title_text='Nifty 50 Historical Close Price', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

st.divider()

# --- FORECASTING MODEL ---
st.header("🔮 Forecast Future Prices")

# Choose the frequency of the prediction
period_type = st.radio(
    "Select Prediction Period Type:", 
    ('Days', 'Weeks', 'Years'), 
    horizontal=True
)

# Dynamically render the slider and calculate total days (period) based on selection
if period_type == 'Days':
    n_days = st.slider('Select number of days to predict:', 1, 90, 30) # Default 30 days, max 90
    period = n_days
    display_period = f"{n_days} Days"
elif period_type == 'Weeks':
    n_weeks = st.slider('Select number of weeks to predict:', 1, 52, 4) # Default 4 weeks, max 52
    period = n_weeks * 7
    display_period = f"{n_weeks} Weeks"
else:
    n_years = st.slider('Select number of years to predict:', 1, 5, 1) # Default 1 year, max 5
    period = n_years * 365
    display_period = f"{n_years} Years"

# Prepare data for Prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
df_train = df_train.dropna()

if len(df_train) < 2:
    st.error("🚨 **Data Error:** Not enough data was fetched from Yahoo Finance to make a prediction. This usually happens if Yahoo Finance is temporarily blocking cloud servers. Please try again in a few minutes.")
    st.stop() 

# Initialize and fit the model
with st.spinner("Training predictive model..."):
    m = Prophet(daily_seasonality=False)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

# --- DISPLAY FORECAST CHARTS ---
st.subheader(f"Forecast Plots ({display_period})")

fig1 = plot_plotly(m, forecast)
fig1.layout.update(title_text=f'Nifty 50 Price Forecast for the next {display_period}', xaxis_rangeslider_visible=True)
st.plotly_chart(fig1, use_container_width=True)

with st.expander("Show Forecast Components (Trends & Seasonality)"):
    st.write("This shows the underlying trends and yearly/weekly seasonality found by the model.")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

st.divider()

# --- TEXT EXPLANATION & INSIGHTS ---
st.header("📝 Forecast Summary & Insights")

# Extracting key data points for the text explanation
last_actual_date = df_train['ds'].iloc[-1].strftime("%Y-%m-%d")
last_actual_price = df_train['y'].iloc[-1]

final_pred_date = forecast['ds'].iloc[-1].strftime("%Y-%m-%d")
final_pred_price = forecast['yhat'].iloc[-1]
final_pred_lower = forecast['yhat_lower'].iloc[-1]
final_pred_upper = forecast['yhat_upper'].iloc[-1]

# Calculate expected percentage change
price_diff = final_pred_price - last_actual_price
percent_change = (price_diff / last_actual_price) * 100
trend = "bullish 📈 (upward)" if price_diff > 0 else "bearish 📉 (downward)"

# Generate plain-english text
summary_text = f"""
Based on the historical data ending on **{last_actual_date}**, the Nifty 50 closed at roughly **{last_actual_price:,.2f}**. 

The AI model projects a **{trend}** trend over the next {display_period}. 
By **{final_pred_date}**, the model predicts the Nifty 50 could reach around **{final_pred_price:,.2f}**, which represents an estimated change of **{percent_change:.2f}%** from the last actual closing price.

**Expected Price Range by {final_pred_date}:**
* **Pessimistic Estimate (Lower Bound):** {final_pred_lower:,.2f}
* **Optimistic Estimate (Upper Bound):** {final_pred_upper:,.2f}

*Keep in mind that as you predict further into the future, the gap between the upper and lower bounds widens because uncertainty increases.*
"""

st.info(summary_text)

# --- DOWNLOAD OPTIONS ---
st.subheader("📥 Download Reports")
st.write("Export the predicted data or the text summary for offline viewing.")

col1, col2 = st.columns(2)

# 1. Download CSV Data
csv_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
    'ds': 'Date', 
    'yhat': 'Predicted_Price', 
    'yhat_lower': 'Lower_Bound', 
    'yhat_upper': 'Upper_Bound'
})
csv_file = csv_data.to_csv(index=False).encode('utf-8')

with col1:
    st.download_button(
        label="📄 Download Forecast Data (CSV)",
        data=csv_file,
        file_name=f"nifty50_forecast_{display_period.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )

# 2. Download Text Summary
with col2:
    st.download_button(
        label="📝 Download Text Summary (TXT)",
        data=summary_text,
        file_name=f"nifty50_summary_{display_period.replace(' ', '_').lower()}.txt",
        mime="text/plain",
    )
