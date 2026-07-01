import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import math

# --- CONFIGURATION ---
START_DATE = "2020-01-01" 
TODAY = date.today().strftime("%Y-%m-%d")

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
    merged_df = pd.merge(main_data[['Date', 'Close']], compare_data[['Date', 'Close']], on='Date', suffixes=('_Nifty', '_Stock'))
    
    merged_df['Nifty_Pct'] = (merged_df['Close_Nifty'] / merged_df['Close_Nifty'].iloc[0] - 1) * 100
    merged_df['Stock_Pct'] = (merged_df['Close_Stock'] / merged_df['Close_Stock'].iloc[0] - 1) * 100

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Nifty_Pct'], name="Nifty 50 (%)", line=dict(color='#1f77b4', width=2)))
    fig_comp.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['Stock_Pct'], name=f"{compare_stock_name} (%)", line=dict(color='#ff7f0e', width=2)))
    
    fig_comp.layout.update(
        title_text=f'Relative Performance: Nifty 50 vs {compare_stock_name}', 
        yaxis_title="Percentage Change (%)", 
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.info(f"""
    **📊 How to interpret this chart:**
    This chart shows the percentage growth of both assets starting from zero on {START_DATE}. 
    * If the orange line ({compare_stock_name}) is above the blue line (Nifty 50), the individual stock has **outperformed** the broader market index over this timeframe.
    * If the blue line is higher, the broader market was a safer, more profitable bet than the individual stock.
    """)

st.divider()

# ==========================================
# SECTION 2: PROPHET TIME-SERIES FORECAST WITH BACKTESTING
# ==========================================
st.header("🔮 AI Trend Forecast & Accuracy Check")
st.write("Train the model on either current data or a past date to check how accurate its previous predictions turned out to be.")

# Mode selection: Live prediction vs Backtesting
analysis_mode = st.radio("Choose Analysis Mode:", ("Live Mode (Predict from Today into Future)", "Backtest Mode (See how past AI models performed vs Reality)"), horizontal=True)

df_all = main_data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"}).dropna()

if analysis_mode == "Live Mode (Predict from Today into Future)":
    cutoff_date = df_all['ds'].max()
    st.success(f"Running Live Mode: Model is training on all available data up to Today ({cutoff_date.strftime('%Y-%m-%d')}).")
else:
    # Allow user to pick a past point in time (e.g., 3 or 6 months ago) to pretend "Today" was back then
    min_cutoff = df_all['ds'].min() + timedelta(days=365) # at least 1 year of training data needed
    max_cutoff = df_all['ds'].max() - timedelta(days=30)  # at least 30 days of actual results to compare against
    
    cutoff_date_input = st.date_input(
        "Pretend 'Today' was this date in the past:", 
        value=(df_all['ds'].max() - timedelta(days=90)).to_pydatetime(),
        min_value=min_cutoff.to_pydatetime(),
        max_value=max_cutoff.to_pydatetime()
    )
    cutoff_date = pd.to_datetime(cutoff_date_input)
    st.info(f"Running Backtest Mode: The AI will hide all data after **{cutoff_date.strftime('%Y-%m-%d')}**, make a blind prediction, and compare it with what actually happened.")

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

# Filter training data to simulate the past cutoff date
df_train = df_all[df_all['ds'] <= cutoff_date]
# Realized data that actually happened after the cutoff date
df_realized_future = df_all[(df_all['ds'] > cutoff_date) & (df_all['ds'] <= cutoff_date + timedelta(days=period))]

with st.spinner("Training predictive AI model..."):
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

# Customizing the Prophet chart for instant readability
fig_forecast = go.Figure()

# Past Historical Training Data (up to cutoff)
fig_forecast.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='markers', name='Data Known by AI', marker=dict(color='black', size=3)))

# Split prediction data 
past_pred = forecast[forecast['ds'] <= cutoff_date]
future_pred = forecast[forecast['ds'] > cutoff_date]

# Uncertainty intervals
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 123, 255, 0.15)', name='Confidence Interval Range'))

# Predicted Trend Lines
fig_forecast.add_trace(go.Scatter(x=past_pred['ds'], y=past_pred['yhat'], mode='lines', line=dict(color='blue', width=1.5), name='AI Historical Fit'))
fig_forecast.add_trace(go.Scatter(x=future_pred['ds'], y=future_pred['yhat'], mode='lines', line=dict(color='red', width=3), name='AI Future Prediction Line'))

# IF BACKTEST MODE: Overlay the REAL numbers that happened after the cutoff date so we can grade the AI
if analysis_mode != "Live Mode (Predict from Today into Future)" and not df_realized_future.empty:
    fig_forecast.add_trace(go.Scatter(x=df_realized_future['ds'], y=df_realized_future['y'], mode='lines+markers', line=dict(color='#ef553b', width=2, dash='dot'), marker=dict(size=4), name='What Actually Happened (Real Results)'))

# Vertical separator line for Cutoff Date
fig_forecast.add_vline(x=cutoff_date, line_width=2, line_dash="dash", line_color="green", annotation_text="Simulation Start Line")

fig_forecast.layout.update(
    title_text=f"Nifty 50 Price Performance vs AI Predictions (Base Date: {cutoff_date.strftime('%Y-%m-%d')})",
    xaxis_title="Date",
    yaxis_title="Price (INR)",
    xaxis_rangeslider_visible=False,
    template="plotly_white"
)
st.plotly_chart(fig_forecast, use_container_width=True)

# Performance calculation metrics
last_actual = df_train['y'].iloc[-1]
predicted_future = future_pred['yhat'].iloc[-1] if not future_pred.empty else last_actual

col1, col2, col3 = st.columns(3)
col1.metric("Price at Simulation Start", f"₹{last_actual:,.2f}")
col2.metric("AI Predicted Target Price", f"₹{predicted_future:,.2f}")

if analysis_mode != "Live Mode (Predict from Today into Future)" and not df_realized_future.empty:
    actual_end_price = df_realized_future['y'].iloc[-1]
    error_pct = ((predicted_future - actual_end_price) / actual_end_price) * 100
    col3.metric("Real Final Price (Difference)", f"₹{actual_end_price:,.2f}", f"AI Error: {error_pct:+.2f}%", delta_color="inverse")
else:
    pct_change_forecast = ((predicted_future - last_actual) / last_actual) * 100
    col3.metric("Expected AI Trend Move", f"{pct_change_forecast:+.2f}%")

st.info("""
**📊 How to use Backtest Mode:**
1. Switch the mode to **Backtest Mode**.
2. Select a date from a few months ago using the date selector tool. 
3. Look at the right side of the **Green Dashed Line**. The **Red Line** shows what the AI guessed would happen, while the **Orange Dotted Line** shows what *actually* happened in the real stock market.
4. Check the third metric card to see the precise calculation error percentage!
""")

st.divider()

# ==========================================
# SECTION 3: RE-ENGINEERED STREAMLINED BINOMIAL TREE
# ==========================================
st.header("🌳 Visual Binomial Probability Tree")
st.write("""
Instead of a single line forecast, financial institutions use a **Binomial Tree Matrix** based on standard deviations to look at best, worst, and expected average path scenarios.
""")

tree_steps = st.slider("Select depth of simulation steps (Nodes):", 2, 15, 6)

# Calculate Volatility from historical data (using data up to the chosen cutoff date for consistency)
main_data_filtered = main_data[main_data['Date'] <= cutoff_date].copy()
main_data_filtered['Daily_Return'] = main_data_filtered['Close'].pct_change()
daily_volatility = main_data_filtered['Daily_Return'].std()
annual_volatility = daily_volatility * math.sqrt(252)

tree_last_price = main_data_filtered['Close'].iloc[-1]
T = 0.5 
dt = T / tree_steps

u = math.exp(annual_volatility * math.sqrt(dt))
d = 1 / u

edge_x = []
edge_y = []
node_x = []
node_y = []
node_text = []

for i in range(tree_steps + 1):
    for j in range(i + 1):
        p = tree_last_price * (u ** (i - j)) * (d ** j)
        node_x.append(i)
        node_y.append(p)
        node_text.append(f"Step {i}, Node {j}<br>Simulated Price: ₹{p:,.2f}")
        
        if i < tree_steps:
            edge_x.extend([i, i + 1, None])
            edge_y.extend([p, p * u, None])
            edge_x.extend([i, i + 1, None])
            edge_y.extend([p, p * d, None])

fig_tree = go.Figure()
fig_tree.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='rgba(150,150,150,0.4)', width=1.5), hoverinfo='none', showlegend=False))
fig_tree.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=9, color='#2ca02c', line=dict(color='white', width=1)), text=node_text, hoverinfo='text', name='Price Nodes'))

best_case = tree_last_price * (u ** tree_steps)
worst_case = tree_last_price * (d ** tree_steps)

fig_tree.layout.update(
    title_text=f"Streamlined Price Matrix Grid (Starting from ₹{tree_last_price:,.2f})",
    xaxis_title="Simulation Steps Forward",
    yaxis_title="Simulated Price Alternatives (INR)",
    template="plotly_white",
    hovermode='closest'
)

st.plotly_chart(fig_tree, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Extreme Bull Case (Top Node)", f"₹{best_case:,.2f}", f"+{((best_case-tree_last_price)/tree_last_price)*100:.1f}%")
c2.metric("Historical Annual Volatility", f"{annual_volatility*100:.2f}%")
c3.metric("Extreme Bear Case (Bottom Node)", f"₹{worst_case:,.2f}", f"{((worst_case-tree_last_price)/tree_last_price)*100:.1f}%")

st.info(f"""
**📊 How to interpret this chart:**
Hover your mouse over any green circle node to instantly see what price level that specific combination of market jumps generates.
* The **Top Boundary Nodes** demonstrate a market where positive volatility hits repeatedly.
* The **Bottom Boundary Nodes** demonstrate continuous down-cycles.
* The dense clustering of nodes in the **Middle** represents the highest statistical area of probability based on the Nifty 50's current historical volatility indicator.
""")
