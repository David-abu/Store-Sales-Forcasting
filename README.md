# Store-Sales-Forcasting
A sales analytics using regression model

## Project Option 3: Store Sales Forecasting (Time Series)
## Objective: Time Series Forecasting

## Description: Forecast store sales by product family (one level of hierarchy). You can forecast sales for different product families separately, or aggregate to overall sales. This allows for manageable complexity while still working with real retail data.

## Data Link:

Kaggle Competition: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

## How to Download Data:

This repo includes `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`, and other small files in `./data/`. **`train.csv` is not on GitHub** (over 100 MB). After cloning, get the full dataset as follows:

1. **Install Kaggle API:** `pip install kaggle`
2. **Set up credentials:** Place your `kaggle.json` (from [Kaggle Account → API](https://www.kaggle.com/settings)) in `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`).
3. **Download and unzip** (from the project root):
   ```bash
   kaggle competitions download -c store-sales-time-series-forecasting -p ./data
   cd data && unzip store-sales-time-series-forecasting.zip && cd ..
   ```
   Or in PowerShell:
   ```powershell
   kaggle competitions download -c store-sales-time-series-forecasting -p ./data
   Expand-Archive -Path .\data\store-sales-time-series-forecasting.zip -DestinationPath .\data -Force
   ```
4. **Optional – Python:** `api.competition_download_files('store-sales-time-series-forecasting', path='./data')` (then unzip the downloaded file in `./data`).

## Key Tasks:

1. Load and explore the store sales data
2. Aggregate sales by product family (one level of hierarchy) or use overall sales
3. Create time series at daily/weekly level
4. Forecast 1-month ahead (30 days) sales using ARIMA, ETS, Prophet models
5. Handle seasonalities (weekly, monthly, yearly patterns)
6. Compare different forecasting models using MAE, RMSE, MAPE
7. Provide forecast visualizations and confidence intervals

Note: You can forecast by product family (e.g., GROCERY I, BEVERAGES, etc.) or aggregate to total sales - choose one approach
Forecast Horizon: 1 month (30 days) ahead

---

## Project summary and key findings

This notebook walks through a complete 30‑day‑ahead forecasting project for the Ecuadorian grocery store data. Below is a consolidated summary of the main results and insights from the code cells.

### Data, preprocessing, and repair
- **Data source**: Kaggle *Store Sales – Time Series Forecasting* competition (`train.csv`, `test.csv`, `stores.csv`, `oil.csv`, holidays). We load from `./data` when available, otherwise generate synthetic data for demo.
- **Preprocessing**: We merge store metadata, holidays, and oil, create time features, and build a clean daily total‑sales series plus family/store‑level series with a proper daily frequency.
- **Zero‑run detection and repair**: We scan each product family for long runs (≥ 7 days) of zero sales, log them in `Repair_Periods.csv`, and optionally repair them using Prophet‑based interpolation. This produces a repaired total series (`Total_Repaired_Daily.csv`) that smooths obvious data glitches while preserving genuine seasonality.

### Structural break and stability
- **Pre‑ vs post‑2014**: Visuals and summary statistics show that sales levels and variability change after 2014, so we define a **post‑break total daily series** (`y_total_daily_post` from 2014‑01‑01 onward) and use that as the main target for daily models.
- **Effect of repair**: Comparing raw vs repaired daily totals shows that repairs mainly affect a small number of abnormal periods; the overall level, trend, and weekly seasonality remain similar, so the business conclusions are stable.

### Exploratory data analysis (EDA)
- **Trend and seasonality**: Total sales grow over time with clear **weekly patterns** (higher weekends) and some **annual/holiday seasonality** visible in the seasonal decomposition plots.
- **Day‑of‑week and month effects**: Boxplots confirm systematic differences by **day of week** and **month**, supporting the use of weekly seasonality and holiday features in the models.
- **Promotions**: Daily `onpromotion` counts are positively correlated with sales; scatterplots and time‑series overlays show that big promotion days tend to coincide with sales spikes.
- **Holidays**: Holiday flags line up with noticeable movements in sales (closures or strong peaks), justifying their inclusion as regressors.

### Train / validation setup
- **Horizon**: We evaluate a **30‑day horizon**.
- **Split modes**: The helper `make_train_test` supports (a) **`last_30`**: last 30 days as validation, and (b) **`80_20`**: chronological 80/20 split. The default in most daily‑model sections is `last_30`, matching the project goal.

### Daily models on total sales (post‑break)
We fit three main daily models on `y_total_daily_post` and compare them on the 30‑day validation window.

- **ETS (Holt‑Winters, daily total)**:
  - Additive trend + additive weekly seasonality on the post‑break daily total series.
  - Captures the **overall level and weekly pattern** well, with moderate forecasting error (see ETS plots and metrics table).
- **ARIMA / SARIMA (daily total)**:
  - Seasonal ARIMA with weekly seasonality on the same daily series.
  - Often achieves similar or slightly better error than ETS, but is more sensitive to parameter choices and residual autocorrelation (checked via ACF/PACF of residuals).
- **Prophet (daily total)**:
  - Includes yearly + weekly seasonality and holiday effects (and, in some variants, regressors such as promotions or oil).
  - Typically delivers **competitive MAPE** and good visual alignment with the validation period, especially around holidays and trend shifts.

Overall, **all three models achieve reasonable 30‑day forecasts**, with small differences in MAE/RMSE/MAPE. In many runs, Prophet and SARIMA slightly outperform plain ETS on percentage error, but ETS remains a strong, simple baseline.

### Diagnostics and residual analysis
- **ACF / PACF of residuals**: For the best daily model by MAPE, residual ACF/PACF plots show whether there is remaining autocorrelation. Some low‑lag structure may remain, suggesting potential gains from richer seasonal or exogenous terms.
- **Regression on residuals**: We regress **test residuals** on promotion counts and holiday flags. If coefficients are small or insignificant, it indicates these effects are already well captured; significant coefficients imply remaining systematic structure the model could learn.

### Family‑level and additional EDA
- **Top‑N families with ETS**: We build daily series per product family, fit ETS models to the top‑N families by revenue, and compare error metrics. This highlights which categories are easiest/hardest to forecast and where promotions or holidays matter most.
- **Geography / store‑level EDA**: Store‑level plots show cross‑sectional differences in average sales and volatility across locations, which can motivate store‑specific models or hierarchical approaches in future work.

### Impact of oil
- **Oil as a regressor**: Oil prices are joined at the daily level and can be used as an exogenous regressor (especially in SARIMAX and Prophet). In focused experiments (see `assignment.py`), models with **holidays + oil** are compared to ETS with no exogenous variables.
- **Conclusion on oil**: For this dataset, oil provides **at most modest incremental gains** over strong baselines that already capture trend, weekly seasonality, promotions, and holidays. The main demand drivers remain **internal levers (promotions, assortment, holidays)** and regular seasonality rather than external oil price fluctuations.

### Business takeaway
- **Forecast quality**: The best daily models achieve relatively low MAPE on the last 30 days, indicating good short‑term forecasting performance.
- **Drivers of sales**: Weekly seasonality, promotions, and holidays are the dominant drivers of variation in total sales; structural changes over time require focusing on post‑break periods for stable modeling.
- **Use in practice**: The final models and visuals can support **inventory planning, staffing, and promotion scheduling** over a 30‑day horizon, with the option to extend to family‑ or store‑level decision making using the same framework.
