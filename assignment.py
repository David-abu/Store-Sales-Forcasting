# %%
# %%
# Store Sales – Total Sales Forecasting with Holidays + Oil (SARIMA, ETS, Prophet)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.style.use("seaborn-v0_8")
pd.set_option("display.max_columns", None)

print("Current working directory:", os.getcwd())

# -------------------------------------------------------------------
# 1. LOAD DATA & BUILD TOTAL DAILY SERIES
# -------------------------------------------------------------------
train = pd.read_csv("train.csv", parse_dates=["date"])

daily_total = (
    train.groupby("date")["sales"]
    .sum()
    .reset_index()
    .sort_values("date")
)
daily_total.set_index("date", inplace=True)

print("\nDaily total sales head:")
print(daily_total.head(), "\n")
print("Date range in raw daily series:",
      daily_total.index.min(), "to", daily_total.index.max(), "\n")

# Reindex to daily frequency (this creates NaNs on missing calendar days)
y = daily_total["sales"].asfreq("D")

# Check gaps
full_idx = pd.date_range(y.index.min(), y.index.max(), freq="D")
y_full = y.reindex(full_idx)
y_full.index.name = "date"

missing_days = y_full[y_full.isna()].index
print("Number of missing calendar days:", len(missing_days))
print("Missing dates:", list(missing_days), "\n")

# -------------------------------------------------------------------
# 2. DEFINE HOLIDAYS (CHRISTMAS CLOSURES) AND CLEAN SERIES
# -------------------------------------------------------------------
christmas_days = pd.to_datetime([
    "2013-12-25", "2014-12-25", "2015-12-25", "2016-12-25"
])

# Ensure Christmas days exist in the daily index
y = y_full.copy()

# Holiday dummy: 1 on Christmas, 0 otherwise
holiday_dummy = pd.Series(0, index=y.index, name="christmas")
holiday_dummy.loc[holiday_dummy.index.isin(christmas_days)] = 1

print("Holiday dummy summary (number of 1s):", int(holiday_dummy.sum()), "\n")

# CLEAN SERIES: interpolate missing values (including Christmas)
print("NaNs in y before interpolation:", y.isna().sum())
y_clean = y.interpolate(method="time")
print("NaNs in y after interpolation:", y_clean.isna().sum(), "\n")

# This is the main series we will use
series = y_clean

# -------------------------------------------------------------------
# 3. INCORPORATE OIL AS EXOGENOUS REGRESSOR
# -------------------------------------------------------------------
oil = pd.read_csv("oil.csv", parse_dates=["date"])
oil = oil.sort_values("date").set_index("date")

oil["dcoilwtico"] = oil["dcoilwtico"].astype(float)
oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()

# Align oil to the same index as our series
oil_aligned = oil.reindex(series.index).ffill().bfill()
oil_series = oil_aligned["dcoilwtico"].rename("oil")

print("Oil series head aligned to sales:")
print(pd.concat([series.rename("sales"), oil_series], axis=1).head(), "\n")

# -------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------------
print("Summary statistics for cleaned total daily sales:\n")
print(series.describe(), "\n")

# Time series plot
fig, ax = plt.subplots(figsize=(12, 4))
series.plot(ax=ax)
ax.set_title("Total Daily Sales Over Time (Cleaned)")
ax.set_ylabel("Sales")
plt.show()

# Distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(series, bins=40, ax=ax[0])
ax[0].set_title("Distribution of Daily Sales (Cleaned)")
sns.boxplot(x=series, ax=ax[1])
ax[1].set_title("Boxplot of Daily Sales (Cleaned)")
plt.tight_layout()
plt.show()

# Day-of-week and month patterns
df = series.to_frame(name="sales")
df["dow"] = df.index.dayofweek  # 0=Mon,...,6=Sun
df["month"] = df.index.month
df["year"] = df.index.year

plt.figure(figsize=(10, 4))
sns.boxplot(x="dow", y="sales", data=df)
plt.title("Sales by Day of Week (Cleaned)")
plt.xlabel("Day of Week (0=Mon)")
plt.show()

plt.figure(figsize=(10, 4))
sns.boxplot(x="month", y="sales", data=df)
plt.title("Sales by Month (Cleaned)")
plt.xlabel("Month")
plt.show()

# Seasonal decomposition (weekly)
decomp = seasonal_decompose(series, model="additive", period=7)
decomp.plot()
plt.suptitle("Seasonal Decomposition (Weekly Period) – Cleaned Series", y=1.02)
plt.show()

# -------------------------------------------------------------------
# 5. TRAIN / TEST SPLIT (30-DAY HORIZON)
# -------------------------------------------------------------------
horizon = 30

y_train = series.iloc[:-horizon]
y_test = series.iloc[-horizon:]

exog_holiday = holiday_dummy
exog_holiday_train = exog_holiday.loc[y_train.index]
exog_holiday_test = exog_holiday.loc[y_test.index]

exog_oil = oil_series
exog_oil_train = exog_oil.loc[y_train.index]
exog_oil_test = exog_oil.loc[y_test.index]

print("Training period:", y_train.index.min(), "to", y_train.index.max())
print("Test (forecast) period:", y_test.index.min(), "to", y_test.index.max(), "\n")

# -------------------------------------------------------------------
# 6. EVALUATION FUNCTION (MAE, RMSE, MAPE)
# -------------------------------------------------------------------
def evaluate_forecast(y_true, y_pred, model_name):
    y_true = pd.Series(y_true, index=y_true.index)
    y_pred = pd.Series(y_pred, index=y_true.index)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)     # older sklearn: no 'squared' arg
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"{model_name} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    return {"model": model_name, "MAE": mae, "RMSE": rmse, "MAPE": mape}

results = []

# -------------------------------------------------------------------
# 7. SARIMA (SARIMAX) WITH HOLIDAYS + OIL
# -------------------------------------------------------------------
print("Fitting SARIMA (SARIMAX) with holiday + oil exog...")

y_train_log = np.log1p(y_train)

exog_train_sarima = pd.concat(
    [exog_holiday_train.rename("christmas"), exog_oil_train.rename("oil")],
    axis=1
)
exog_test_sarima = pd.concat(
    [exog_holiday_test.rename("christmas"), exog_oil_test.rename("oil")],
    axis=1
)

order = (1, 1, 1)
seasonal_order = (1, 1, 1, 7)  # weekly seasonality

sarima_model = sm.tsa.SARIMAX(
    y_train_log,
    exog=exog_train_sarima,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
)

sarima_res = sarima_model.fit(disp=False)
print(sarima_res.summary())

sarima_forecast_log = sarima_res.forecast(steps=horizon, exog=exog_test_sarima)
sarima_forecast = np.expm1(sarima_forecast_log)
sarima_forecast.index = y_test.index

results.append(evaluate_forecast(y_test, sarima_forecast, "SARIMA (holidays + oil)"))

plt.figure(figsize=(12, 4))
plt.plot(y_train.index, y_train, label="Train")
plt.plot(y_test.index, y_test, label="Test", color="black")
plt.plot(sarima_forecast.index, sarima_forecast, label="SARIMA Forecast")
plt.title("SARIMA 30-Day Forecast (with Holidays + Oil)")
plt.legend()
plt.show()

# -------------------------------------------------------------------
# 8. ETS (Holt-Winters) – NO EXOG
# -------------------------------------------------------------------
print("\nFitting ETS (Holt-Winters) model (no exog)...")

ets_model = ExponentialSmoothing(
    y_train,
    trend="add",
    seasonal="add",        # additive for robustness if any small/zero values
    seasonal_periods=7
)
ets_res = ets_model.fit()
ets_forecast = ets_res.forecast(horizon)
ets_forecast.index = y_test.index

results.append(evaluate_forecast(y_test, ets_forecast, "ETS (no exog)"))

plt.figure(figsize=(12, 4))
plt.plot(y_train.index, y_train, label="Train")
plt.plot(y_test.index, y_test, label="Test", color="black")
plt.plot(ets_forecast.index, ets_forecast, label="ETS Forecast")
plt.title("ETS 30-Day Forecast")
plt.legend()
plt.show()

# -------------------------------------------------------------------
# 9. PROPHET WITH HOLIDAYS + OIL
# -------------------------------------------------------------------
print("\nFitting Prophet with Christmas holidays + oil regressor...")

df_prophet_train = y_train.reset_index()
df_prophet_train.columns = ["ds", "y"]
df_prophet_train["oil"] = exog_oil_train.values

holidays_df = pd.DataFrame({
    "holiday": "christmas_closure",
    "ds": christmas_days,
    "lower_window": 0,
    "upper_window": 0,
})

m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    holidays=holidays_df
)
m.add_regressor("oil")

m.fit(df_prophet_train)

future = m.make_future_dataframe(periods=horizon, freq="D")
future["oil"] = exog_oil.loc[future["ds"]].values

forecast = m.predict(future)

forecast_h = forecast.set_index("ds").loc[y_test.index]
prophet_pred = forecast_h["yhat"]

results.append(evaluate_forecast(y_test, prophet_pred, "Prophet (holidays + oil)"))

m.plot(forecast)
plt.title("Prophet Forecast (with Holidays + Oil)")
plt.show()

m.plot_components(forecast)
plt.show()

m.plot(forecast.loc[forecast["ds"].between(y_test.index.min(), y_test.index.max())])
plt.title("Prophet 30-Day Forecast Horizon (with Holidays + Oil)")
plt.show()

# -------------------------------------------------------------------
# 10. MODEL COMPARISON & SAVING RESULTS
# -------------------------------------------------------------------
results_df = pd.DataFrame(results)
print("\nModel comparison (lower is better):")
print(results_df)

plt.figure(figsize=(12, 4))
plt.plot(y_test.index, y_test, label="Actual", color="black")
plt.plot(sarima_forecast.index, sarima_forecast, label="SARIMA (holidays + oil)")
plt.plot(ets_forecast.index, ets_forecast, label="ETS (no exog)")
plt.plot(prophet_pred.index, prophet_pred, label="Prophet (holidays + oil)")
plt.title("30-Day Forecast Comparison")
plt.legend()
plt.show()

# Save model metrics
results_df.to_csv("model_results_total_sales_with_oil.csv", index=False)

# Save forecasts with actuals for the test period
out = pd.DataFrame({
    "date": y_test.index,
    "actual": y_test.values,
    "sarima_holiday_oil": sarima_forecast.values,
    "ets_no_exog": ets_forecast.values,
    "prophet_holiday_oil": prophet_pred.values,
})
out.to_csv("forecasts_total_sales_with_oil.csv", index=False)

print("Saved model_results_total_sales_with_oil.csv and forecasts_total_sales_with_oil.csv")



