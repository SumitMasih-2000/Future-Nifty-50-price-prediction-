import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import math

# --- CONFIGURATION ---
START_DATE = "2020-01-01" # Shortened for faster loading and more relevant recent volatility
TODAY = date.today().strftime("%Y-%m-%d")

# A sample of Top Nifty 50 companies for comparison
NIFTY_50_TICKERS = {
    "Nifty 50 Index": "^NSEI",
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS"
}

st.set_page_config(page_title="Advanced Nifty 50 Analyst", layout="wide")

# --- UI HEADER ---
st.title("📈 Advanced Nifty 50 Analyst & Predictor")

st.warning("""
**🚨 IMPORTANT ALERT:** The stock market is highly sensitive to real-world events. The predictions and models shown here represent mathematical probabilities based on past data, **not** guaranteed true future prices. Use this for educational analysis, not financial advice.
""")

# --- DATA FETCHING ---
@st.cache_data
def load_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(start=START_DATE, end=TODAY)
    data.reset_index(inplace=True)
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
    return data

# Load default Nifty 50 data
main_data = load_data("^NSEI")

st.divider()

# ==========================================
# SECTION 1: COMPARATIVE ANALYSIS
# ==========================================
st.header("⚖️ Comparative Analysis")
st.write("Compare the baseline Nifty 50 index against major individual constituent stocks to see which is outperforming.")

compare_stock_name = st.selectbox("Select a Nifty 50 Stock to compare with the Nifty 50 Index:", list(NIFTY_50_TICKERS.keys())[1:])
compare_ticker = NIFTY_50_TICKERS[compare_stock_name]

compare_data = load_data(compare_ticker)

if not main_data.empty and not compare_data.empty:
    # Merge data on Date to ensure apples-to-apples comparison
    merged_df = pd.merge(main_data[['Date', 'Close']], compare_data[['Date', 'Close']], on='Date', suffixes=('_Nifty', '_Stock'))
    
    # Calculate percentage change from the first day in the dataset
    merged_df['Nifty_Pct'] = (merged_df['Close_Nifty'] / merged_df['Close_Nifty'].iloc[0] - 1) * 100
    merged_df['Stock_Pct'] = (merged_df['Close_Stock'] / merged_df['Close_Stock'].iloc[0] - 1) * 100

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Nifty_Pct'], name="Nifty 50 (%)", line=dict(color='blue')))
    fig_comp.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Stock_Pct'], name=f"{compare_stock_name} (%)", line=dict(color='orange')))
    fig_comp.layout.update(title_text=f'Relative Performance: Nifty 50 vs {compare_stock_name}', yaxis_title="Percentage Change (%)", xaxis_rangeslider_visible=True)
    
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Interpretation
    st.info(f"""
    **📊 How to interpret this chart:**
    This chart shows the percentage growth of both assets starting from zero on {START_DATE}. 
    * If the orange line ({compare_stock_name}) is above the blue line (Nifty 50), the individual stock has **outperformed** the broader market index over this timeframe.
    * If the blue line is higher, the broader market was a safer, more profitable bet than the individual stock.
    """)

st.divider()

# ==========================================
# SECTION 2: PROPHET TIME-SERIES FORECAST
# ==========================================
st.header("🔮 AI Trend Forecast (Prophet)")

period_type = st.radio("Select Forecast Period Type:", ('Days', 'Weeks', 'Years'), horizontal=True)

if period_type == 'Days':
    n_days = st.slider('Select number of days to predict:', 1, 90, 30)
    period = n_days
    display_period = f"{n_days} Days"
elif period_type == 'Weeks':
    n_weeks = st.slider('Select number of weeks to predict:', 1, 52, 4)
    period = n_weeks * 7
    display_period = f"{n_weeks} Weeks"
else:
    n_years = st.slider('Select number of years to predict:', 1, 5, 1)
    period = n_years * 365
    display_period = f"{n_years} Years"

df_train = main_data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"}).dropna()

with st.spinner("Training predictive AI model..."):
    m = Prophet(daily_seasonality=False)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

fig_forecast = plot_plotly(m, forecast)
fig_forecast.layout.update(title_text=f'Nifty 50 Price Forecast for the next {display_period}', xaxis_rangeslider_visible=True)
st.plotly_chart(fig_forecast, use_container_width=True)

st.info(f"""
**📊 How to interpret this chart:**
* The **Black Dots** represent actual historical closing prices.
* The **Blue Line** is the AI's predicted average trend line.
* The **Light Blue Shaded Area** represents the "Confidence Interval." The wider this area gets, the more uncertain the AI is about the exact price. This proves that there is no single "true" future price, only a mathematically probable range.
""")

st.divider()

# ==========================================
# SECTION 3: BINOMIAL OPTIONS/PRICE TREE
# ==========================================
st.header("🌳 Binomial Price Tree Simulation")
st.write("""
Instead of a single AI prediction, financial engineers use Binomial Trees based on **historical volatility** to map out possible future price paths step-by-step.
""")

tree_steps = st.slider("Select number of future steps (Nodes):", 2, 10, 5)

# Calculate Volatility from historical data
main_data['Daily_Return'] = main_data['Close'].pct_change()
daily_volatility = main_data['Daily_Return'].std()
annual_volatility = daily_volatility * math.sqrt(252) # 252 trading days in a year

last_price = main_data['Close'].iloc[-1]
T = 1.0 # 1 year forward looking (normalized)
dt = T / tree_steps

# Binomial factors (Cox-Ross-Rubinstein model)
u = math.exp(annual_volatility * math.sqrt(dt))
d = 1 / u

# Generate tree nodes
tree_fig = go.Figure()

for step in range(tree_steps):
    for node in range(step + 1):
        # Current node price
        current_node_price = last_price * (u ** (step - node)) * (d ** node)
        
        # Calculate next step prices (Up and Down)
        price_up = current_node_price * u
        price_down = current_node_price * d
        
        # Draw line going UP
        tree_fig.add_trace(go.Scatter(
            x=[step, step + 1], y=[current_node_price, price_up],
            mode='lines+markers', line=dict(color='green', width=2),
            marker=dict(size=8, color='black'), hoverinfo='y', showlegend=False
        ))
        
        # Draw line going DOWN
        tree_fig.add_trace(go.Scatter(
            x=[step, step + 1], y=[current_node_price, price_down],
            mode='lines+markers', line=dict(color='red', width=2),
            marker=dict(size=8, color='black'), hoverinfo='y', showlegend=False
        ))

tree_fig.layout.update(
    title_text=f"Binomial Price Path Simulation ({tree_steps} Steps)",
    xaxis_title="Simulation Steps (Forward in Time)",
    yaxis_title="Simulated Price Level",
    showlegend=False
)

st.plotly_chart(tree_fig, use_container_width=True)

# Math and Interpretation
st.info(f"""
**📊 How to interpret this chart:**
This tree takes the last closing price (**{last_price:,.2f}**) and projects it forward using the Nifty 50's historical annualized volatility (**{annual_volatility*100:.2f}%**). 

At each step, the price can either jump **UP** (Green line) by a mathematical factor, or jump **DOWN** (Red line) by a factor. 
* The **top-most node** on the far right represents the absolute best-case scenario if the market goes up every single step.
* The **bottom-most node** represents the absolute worst-case scenario.
* The nodes in the middle represent the most statistically probable outcomes where the market fluctuates up and down.

*Geeky details: This uses the Cox-Ross-Rubinstein formulas where the up factor is calculated as $u = e^{{\sigma \sqrt{{\Delta t}}}}$ and the down factor is $d = 1/u$.*
""")
