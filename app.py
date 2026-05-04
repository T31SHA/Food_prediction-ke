import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Kenya Maize Price Forecast", layout="wide")

# Title and description
st.title("🌽 Kenya Maize (White) Price Forecasting")
st.markdown("""
**CRISP-DM based analysis** | SARIMA Model for Monthly National Average Prices  
*Helping with food security tracking and economic planning in Kenya*
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('wfp_food_prices_ken.csv')
    return df

df = load_data()

# Data Preparation (same as notebook)
@st.cache_data
def prepare_maize_data(df):
    maize_df = df[df['commodity'] == 'Maize (white)'].copy()
    maize_df['date'] = pd.to_datetime(maize_df['date'])
    
    # Monthly average
    monthly_price = maize_df.groupby(maize_df['date'].dt.to_period('M'))['price'].mean().reset_index()
    monthly_price['date'] = monthly_price['date'].dt.to_timestamp()
    monthly_price.set_index('date', inplace=True)
    
    # Full range + linear interpolation
    full_range = pd.date_range(start=monthly_price.index.min(), end=monthly_price.index.max(), freq='MS')
    monthly_price = monthly_price.reindex(full_range)
    monthly_price['price'] = monthly_price['price'].interpolate(method='linear')
    
    return monthly_price

monthly_price = prepare_maize_data(df)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Go to", 
    ["📊 Data Overview", "🔮 SARIMA Forecast", "📈 Model Evaluation", "📋 Methodology & Insights"])

# ==================== PAGE 1: DATA OVERVIEW ====================
if page == "📊 Data Overview":
    st.header("1. Business & Data Understanding")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Time Range", f"{df['date'].min()} to {df['date'].max()}")
    
    with col2:
        st.metric("Unique Commodities", df['commodity'].nunique())
        st.metric("Markets Covered", df['market'].nunique())
    
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Maize Price Time Series (National Monthly Average)")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(monthly_price.index, monthly_price['price'], color='blue', linewidth=2)
    ax.set_title('Monthly Average Maize (White) Price in Kenya (KES)')
    ax.set_ylabel('Price (KES per unit)')
    ax.set_xlabel('Date')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ==================== PAGE 2: FORECAST ====================
elif page == "🔮 SARIMA Forecast":
    st.header("SARIMA Forecast")
    
    train_size = int(len(monthly_price) * 0.8)
    train = monthly_price.iloc[:train_size]
    test = monthly_price.iloc[train_size:]
    
    # Fit model (cached)
    @st.cache_resource
    def fit_model():
        model = SARIMAX(train['price'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
        return model.fit(disp=False)
    
    results = fit_model()
    
    # Forecast
    forecast = results.predict(start=len(train), end=len(monthly_price)-1, dynamic=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train.index, train['price'], label='Training Data', color='blue')
    ax.plot(test.index, test['price'], label='Actual (Test)', color='green')
    ax.plot(test.index, forecast, label='SARIMA Forecast', color='red', linestyle='--')
    ax.set_title('SARIMA Forecast vs Actual Maize Prices')
    ax.set_ylabel('Price (KES)')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.info("**Model Parameters**: SARIMAX(1,1,1)(1,1,0)[12]")

# ==================== PAGE 3: EVALUATION ====================
elif page == "📈 Model Evaluation":
    st.header("Model Performance")
    
    train_size = int(len(monthly_price) * 0.8)
    train = monthly_price.iloc[:train_size]
    test = monthly_price.iloc[train_size:]
    
    results = fit_model()  # from cache
    preds = results.predict(start=len(train), end=len(monthly_price)-1, dynamic=False)
    
    mae = mean_absolute_error(test['price'], preds)
    rmse = np.sqrt(mean_squared_error(test['price'], preds))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f} KES")
    col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f} KES")
    col3.metric("Test Period Length", len(test))
    
    st.warning("The model shows moderate accuracy but struggles with sharp volatility spikes.")

# ==================== PAGE 4: METHODOLOGY & INSIGHTS ====================
else:
    st.header("Methodology, Limitations & Recommendations")
    
    st.subheader("CRISP-DM Process Used")
    st.markdown("""
    - **Business Understanding**: Predict maize prices for food security
    - **Data Preparation**: Filter → Monthly aggregation → Linear interpolation
    - **Modeling**: SARIMA (captures trend + 12-month seasonality)
    - **Evaluation**: 80/20 chronological train-test split
    """)
    
    st.subheader("Key Limitations")
    limitations = """
    - Single variable (price only) — no external regressors (rainfall, inflation, fuel, etc.)
    - Linear interpolation of missing months smooths real shocks
    - National average hides regional differences
    - Single SARIMA configuration tested
    """
    st.info(limitations)
    
    st.subheader("Recommendations")
    recs = """
    1. **SARIMAX** with exogenous variables (weather, exchange rates, etc.)
    2. Grid search / Auto-ARIMA for better parameters
    3. Rolling window backtesting
    4. County-level models
    5. Monthly retraining + alert system
    """
    st.success(recs)
    
    st.markdown("---")
    st.caption("Built with Streamlit • SARIMA Model • Data: WFP Kenya Food Prices")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Sharahbiil's Maize Price Dashboard")

