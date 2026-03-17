import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="StockSense AI",   
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
            background-color: #0d0f14;
            color: #e8eaf0;
}

.main { background-color: #0d0f14; }
.stApp { background-color: #0d0f14; }

h1, h2, h3 {
         font-family: 'Space Mono' , monospace;
         color: #00e5ff;
}

.metric-card {
         background: linear-gradient(135deg, #1a1d26, #12151e);
         border: 1px solid #00e5ff33;
         border-radius: 12px;
         padding: 20px;
         text-align: center;
         box-shadow: 0  0   20px  #00e5ff11;
}

.metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #00e5ff;
}

.metric-label {
        font-size: 0.85rem;
        color: #8892a4;
        margin-top: 4px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
}
.winner-badge {
        background: linear-gradient(90deg, #00e5ff22, #00ff8822);
        border: 1px solid #00ff88;
        border-radius: 20px;
        padding: 6px 16px;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #00ff88;
        display: inline-block;
        margin-top: 8px;
}
            
.section-header{
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #8892a4;
        margin-bottom: 8px;
}
            
.disclaimer {
        background: #1a1d26;
        border-left: 3px solid #ff6b6b;
        padding: 12px 16px;
        border-radius: 0  8px 8px 0;
        font-size: 0.82rem;
        color: #8892a4;
        margin-top: 24px;
}

div[data-testid="stSidebar"] {
         background-color: #12151e;
         border-right: 1px solid #1e2130;
}

.stButton > button {
         background: linear-gradient(90deg, #00e5ff, #00b8d4);
         color: #0d0f14;
         font-family: 'Space Mono' , monospace;
         font-weight: 700;
         border: none;
         border-radius: 8px;
         padding: 10px 24px;
         width: 100%;
         font-size: 0.9rem;
         letter-spacing: 0.05em;
         transition: all 0.2s ease;

stButton > button:hover {
         background: linear-gradient(90deg, #00ff88, #00e5ff):
         transform: translateY(-1px);
}

.stSelectbox label, .stTextInput label, .stSlider label {
         color: #8892a4 !important;
         font-size: 0.82rem !important;
         letter-spacing: 0.08em !important;
         text-transform: uppercase !important;
}

.hero-title {
         font-family: 'Space Mono', monospace;
         font-size: 2.8rem;
         font-weight: 700;
         color: #00e5ff;
         line-height: 1.1;
}
            
.hero-sub {
         font-size: 1rem;
         color: #8892a4;
         margin-top: 8px;
}

hr {
         border-color: #1e2130;
}
</style>
""", unsafe_allow_html = True)

@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
         raw = yf.download(ticker, start=start, end=end, auto_adjust=True)
         if raw.empty:
             return None
         raw.columns = raw.columns.get_level_values(0)
         df = raw[['Close', 'High', 'Low', 'Open', 'Volume']].copy()
         df['Returns'] = df['Close'].pct_change()
         df['Volatility'] = df['Returns'].rolling(30).std()
         df['MA_30'] = df['Close'].rolling(30).mean()
         df['MA_90'] = df['Close'].rolling(90).mean()

         #RSI
         delta = df['Close'].diff()
         gain = delta.where(delta > 0, 0).rolling(14).mean()
         loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
         rs = gain / loss
         df['RSI'] = 100 - (100 / (1 + rs))

         #MACD
         df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        
          
         df.dropna(inplace=True)
         return df

def run_prophet(df, periods=90):
         from prophet import Prophet
         prophet_df = df[['Close']].reset_index()
         prophet_df.columns = ['ds', 'y']
         prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
         prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')   
         prophet_df.dropna(inplace=True)

         model = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                         changepoint_prior_scale=0.05, daily_seasonality=False)
         model.fit(prophet_df)
         future = model.make_future_dataframe(periods=periods)
         forecast = model.predict(future)

         merged = forecast[['ds', 'yhat']].merge(prophet_df, on='ds')
         mae = mean_absolute_error(merged['y'], merged['yhat'])
         rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
         return forecast, mae, rmse


def run_moving_average(df, periods=90):
         close = df['Close'].values
         ma_window = 30
         last_ma = df['Close'].rolling(ma_window).mean().iloc[-1]
         forecast_values = [last_ma] * periods

         in_sample = df['Close'].rolling(ma_window).mean().dropna()
         actual = df['Close'].loc[in_sample.index]
         mae = mean_absolute_error(actual, in_sample)
         rmse = np.sqrt(mean_squared_error(actual, in_sample))

         last_date = df.index[-1]
         future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='B')
         forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast_values})
         return forecast_df, mae, rmse


def plot_price_chart(df, ticker):
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor('#0d0f14')
        ax.set_facecolor('#12151e')
    
        ax.plot(df.index, df['Close'], color='#00e5ff', linewidth=1.5, label='Close Price')
        ax.plot(df.index, df['MA_30'], color='#00ff88', linewidth=1, linestyle='--', alpha=0.7, label='MA 30')
        ax.plot(df.index, df['MA_90'], color='#ff6b6b', linewidth=1, linestyle='--', alpha=0.7, label='MA 90')

        ax.set_title(f'{ticker} - Price History', color='#00e5ff',
                     fontfamily='monospace', fontsize=13, pad=12)
        ax.tick_params(colors='#8892a4')
        ax.spines[:].set_color('#1e2130')
        ax.yaxis.label.set_color('#8892a4')
        ax.xaxis.label.set_color('#8892a4')
        ax.legend(facecolor='#12151e', edgecolor='#1e2130', labelcolor='#e8eaf0', fontsize=9)
        ax.grid(color='#1e2130', linestyle='--', linewidth=0.5, alpha=0.6)
        plt.tight_layout()
        return fig

       
def  plot_forecast(df, forecast_df, ticker, model_name):
       fig, ax = plt.subplots(figsize=(14, 5))
       fig.patch.set_facecolor('#0d0f14')
       ax.set_facecolor('#12151e')

       ax.plot(df.index, df['Close'], color='#00e5ff', linewidth=1.5, label='Historical')

       if 'yhat_lower' in forecast_df.columns:
           future = forecast_df[forecast_df['ds'] > df.index[-1]]
           ax.plot(future['ds'], future['yhat'], color='#00ff88', linewidth=2, label=f'{model_name} Forecast')
           ax.fill_between(future['ds'], future['yhat_lower'], future['yhat_upper'],
                           color='#00ff88', alpha=0.15, label='Confidence Band')
       else:
           ax.plot(forecast_df['ds'], forecast_df['yhat'], color='#00ff88',
                   linewidth=2, linestyle='--', label=f'{model_name} Forecast')
       ax.set_title(f'{ticker} - {model_name} Forecast', color='#00e5ff',
                     fontfamily='monospace', fontsize=13, pad=12)
       ax.tick_params(colors='#8892a4')
       ax.spines[:].set_color('#1e2130')
       ax.legend(facecolor='#12151e', edgecolor='#1e2130', labelcolor='#e8eaf0', fontsize=9)
       ax.grid(color='#1e2130', linestyle='--', linewidth=0.5, alpha=0.6)
       plt.tight_layout()
       return fig

def plot_rsi(df):
       fig, ax = plt.subplots(figsize=(14, 3))
       fig.patch.set_facecolor('#0d0f14')
       ax.set_facecolor('#12151e')

       ax.plot(df.index, df['RSI'], color='#a78bfa', linewidth=1.2)
       ax.axhline(70, color='#ff6b6b', linestyle='--', linewidth=0.8, alpha=0.7)
       ax.axhline(30, color='#00ff88', linestyle='--', linewidth=0.8, alpha=0.7)
       ax.fill_between(df.index, df['RSI'], 70, where=(df['RSI'] >= 70), color='#ff6b6b', alpha=0.15)
       ax.fill_between(df.index, df['RSI'], 30, where=(df['RSI'] <= 30), color='#00ff88', alpha=0.15)

       ax.set_title('RSI (14)', color='#00e5ff', fontfamily='monospace', fontsize=11, pad=8)
       ax.set_ylim(0, 100)
       ax.tick_params(colors='#8892a4')
       ax.spines[:].set_color('#1e2130')
       ax.grid(color='#1e2130', linestyle='--', linewidth=0.5, alpha=0.5)
       plt.tight_layout()
       return fig

with st.sidebar:
       st.markdown('<p class="hero-title">*</p>', unsafe_allow_html=True)
       st.markdown('<p class="hero-title">StockSense AI</p', unsafe_allow_html=True)
       st.markdown('<p class="hero-sub">Intelligent stock forecasting powered by ML</p>', unsafe_allow_html=True)
       st.markdown("---")

       st.markdown('<p class="section-header"> Stock Configuration</p>' , unsafe_allow_html=True)

       ticker = st.text_input("Ticker Symbol", value="AAPL", placeholder="e.g AAPL, TSLA, MSFT").upper().strip()

       col1, col2 = st.columns(2)
       with col1:
              start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
       with col2:
              end_date = st.date_input("End date", value=pd.to_datetime("today"))
       st.markdown("---")
       st.markdown('<p class="section-header">Forecast Settings</p', unsafe_allow_html=True)

       forecast_model = st.selectbox(
          "Forecast Model",
          options=["Prophet", "Moving Average", "Both"],
          index=0
        
       )
       forecast_days = st.slider("Forecast Horizon (days)", min_value=30, max_value=365, value=90, step=15)

       st.markdown("---")
       run_btn = st.button("Run Analysis")
       
       st.markdown("""
       <div class="disclaimer">
              <strong>Disclaimer:</strong> This tool is for educational purposes only.
             Not financial advice. Always do your own research.
       </div>
        """, unsafe_allow_html=True)
       


if not run_btn:
       st.markdown("""
       <div style="text-align:center; padding: 80px 20px;">
            <p class="hero-title">StockSense AI </p>
            <p class="hero-sub"> Enter a ticker symbol in the sidebar and click <strong>Run Analysis</strong> to begin.</p>
            <br />
            <p style="color: #8892a4; font_size: 0.85rem;">Supports Prophet forecasting . RSI & MACD indicators . Moving average models</p>
       </div>
       """, unsafe_allow_html=True)

else:
    if not ticker:
             st.error("Please enter a valid ticker symbol.")
             st.stop()

    with st.spinner(f"Fetching data for **{ticker}**..."):
           df = load_data(ticker, str(start_date), str(end_date))

    if df is None or df.empty:
           st.error(f" No data found for **{ticker}**. Please check the ticker symbol.")
           st.stop()

    latest_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    change = latest_close - prev_close
    change_pct = (change / prev_close) * 100
    volatility = df['Volatility'].iloc[-1] * 100
    rsi_val = df['RSI'].iloc[-1]
    total_return = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
    
      

              
                           
    