# Store-Sales-Forcasting
A sales analytics using regression model

Project Option 3: Store Sales Forecasting (Time Series)
Objective: Time Series Forecasting

Description: Forecast store sales by product family (one level of hierarchy). You can forecast sales for different product families separately, or aggregate to overall sales. This allows for manageable complexity while still working with real retail data.

Data Link:

Kaggle Competition: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
How to Download Data:

Install Kaggle API: pip install kaggle
Set up Kaggle credentials (download kaggle.json from your Kaggle account settings)
Download data using one of these methods:
Command line: kaggle competitions download -c store-sales-time-series-forecasting
Python code:
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.competition_download_files('store-sales-time-series-forecasting', path='./data')
Key Tasks:

1. Load and explore the store sales data
2. Aggregate sales by product family (one level of hierarchy) or use overall sales
3. Create time series at daily/weekly level
4. Forecast 1-month ahead (30 days) sales using ARIMA, ETS, Prophet models
5. Handle seasonalities (weekly, monthly, yearly patterns)
6. Compare different forecasting models using MAE, RMSE, MAPE
7. Provide forecast visualizations and confidence intervals
Note: You can forecast by product family (e.g., GROCERY I, BEVERAGES, etc.) or aggregate to total sales - choose one approach
Forecast Horizon: 1 month (30 days) ahead
