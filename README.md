# stocksense-ai

# 📈 StockSense AI

> An end-to-end machine learning web application for stock price forecasting with real-time data and interactive visualizations.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-ff4b4b?style=flat-square&logo=streamlit)
![Prophet](https://img.shields.io/badge/Prophet-forecasting-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🚀 Live Demo

👉 **[Launch StockSense AI](https://stocksense-ai-wsgiyhekzr2jkyrjphqw2t.streamlit.app)**

---

## 📌 Overview

StockSense AI is a professional stock forecasting dashboard that allows users to:

- Input any stock ticker (AAPL, TSLA, NVDA, BTC-USD, etc.)
- Fetch real-time historical price data
- Analyze stocks using technical indicators
- Generate AI-powered price forecasts
- Compare model performance metrics

Built with a dark fintech aesthetic and deployed on Streamlit Community Cloud.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 Price History | Interactive chart with 30 & 90-day moving averages |
| 🤖 Prophet Forecast | Facebook's time series model for future price prediction |
| 📉 Moving Average | Statistical baseline forecast model |
| 🔄 Model Comparison | Side-by-side MAE & RMSE evaluation |
| 📈 RSI Indicator | Overbought/oversold signal (14-day) |
| ⚡ MACD | Momentum and trend direction indicator |
| 💹 KPI Cards | Live price, daily change, volatility, total return |
| 📋 Raw Data | Scrollable historical data table |

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Data:** yfinance (Yahoo Finance API)
- **Forecasting:** Prophet (Meta/Facebook)
- **ML:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Deployment:** Streamlit Community Cloud

---

## ⚙️ Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Santandave961/stocksense-ai.git
cd stocksense-ai
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

Or with Anaconda:
```bash
conda install -c conda-forge prophet
pip install streamlit yfinance pandas numpy matplotlib scikit-learn
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## 📁 Project Structure

```
stocksense-ai/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 📊 How to Use

1. Enter a **stock ticker** in the sidebar (e.g. `AAPL`, `TSLA`, `NVDA`)
2. Set your **date range** (default: 2020 to today)
3. Choose a **forecast model** (Prophet, Moving Average, or Both)
4. Set **forecast horizon** in days (30–365)
5. Click **🚀 Run Analysis**

---

## 🧠 Models Used

### Prophet
- Developed by Meta (Facebook)
- Handles seasonality and trend changes
- Outputs confidence intervals
- Best for long-term forecasting

### Moving Average
- 30-day rolling average baseline
- Simple and interpretable
- Used for model comparison benchmark

### Evaluation Metrics
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error

---

## ⚠️ Disclaimer

> This application is built for **educational and portfolio purposes only**.
> It is **not financial advice**. Always conduct your own research before
> making any investment decisions.

---

## 👤 Author

**Okparaji Wisdom**
- GitHub: [@Santandave961](https://github.com/Santandave961)

---

## 📄 License

This project is licensed under the MIT License.

---

⭐ **If you found this project useful, please give it a star!**
