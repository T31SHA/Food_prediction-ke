```python?code_reference&code_event_index=5
readme_content = """# 🌾 Kenyan Food Price Forecaster

> **Predictive modeling for food security and economic planning.**

An end-to-end time series forecasting pipeline predicting the future market prices of essential commodities (e.g., White Maize) in Kenya. Built on the **CRISP-DM** framework.

##  Methodology

- **Business Understanding:** Mitigating food insecurity via predictive price tracking.
- **Data Preparation:** Temporal aggregation, missing value interpolation, and univariate structuring.
- **Modeling:** Seasonal Autoregressive Integrated Moving Average (**SARIMA**).
- **Evaluation:** Baseline performance measured via MAE and RMSE on a 80/20 sequential split.

## 🛠 Tech Stack

- `pandas` / `numpy` — Data manipulation
- `statsmodels` — Time-series modeling (SARIMA)
- `scikit-learn` — Model evaluation metrics
- `matplotlib` — Forecasting visualization

##  Quick Start

1. **Clone & Install:**
