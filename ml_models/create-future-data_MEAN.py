import pandas as pd
import numpy as np
import os
import joblib

# Path for prediction data
CURR_DIR_PATH = os.getcwd()
DATA_PATH = CURR_DIR_PATH + "/data/clean-data/"
# Ski Centers: Salla.csv, Kommattivaara.csv, Ruskotunturi.csv, Himos.csv, Salpausselkä.csv
SKI_CENTER_PATH = DATA_PATH + "Salla.csv" # Change the .csv in quotes as needed

# Read data to extrapolate & Transform date
pred_df = pd.read_csv(SKI_CENTER_PATH)

pred_df['Date'] = pd.to_datetime(pred_df['Date'])
pred_df['DayOfYear'] = pred_df['Date'].dt.dayofyear
pred_df['Year'] = pred_df['Date'].dt.year

# Calculate daily mean values from the data
daily_means = pred_df.groupby(['DayOfYear', 'Year'])[['Cloud_value', 'Average temperature [°C]', 'Solar Radiation (kW-hr/m^2/day)', 'Latitude']].mean()

# Generate future data DataFrame with date range
future_dates = pd.date_range(start='2024-01-01', end='2050-12-31', freq='D')
future_data = pd.DataFrame({'Date': future_dates})

future_data['DayOfYear'] = future_data['Date'].dt.dayofyear
future_data['Year'] = future_data['Date'].dt.year

# Merge future data with daily means to get corresponding values
future_data = future_data.merge(daily_means, on=['DayOfYear', 'Year'], how='left')

# Fill missing values with overall means
overall_means = pred_df[['Cloud_value', 'Average temperature [°C]', 'Solar Radiation (kW-hr/m^2/day)', 'Latitude']].mean()
future_data.fillna(overall_means, inplace=True)

# Clip values using min and max from pred_df
min_values = pred_df[['Cloud_value', 'Average temperature [°C]', 'Solar Radiation (kW-hr/m^2/day)', 'Latitude']].min()
max_values = pred_df[['Cloud_value', 'Average temperature [°C]', 'Solar Radiation (kW-hr/m^2/day)', 'Latitude']].max()

future_data['Cloud_value'] = future_data['Cloud_value'].clip(min_values['Cloud_value'], max_values['Cloud_value'])
future_data['Average temperature [°C]'] = future_data['Average temperature [°C]'].clip(min_values['Average temperature [°C]'], max_values['Average temperature [°C]'])
future_data['Solar Radiation (kW-hr/m^2/day)'] = future_data['Solar Radiation (kW-hr/m^2/day)'].clip(min_values['Solar Radiation (kW-hr/m^2/day)'], max_values['Solar Radiation (kW-hr/m^2/day)'])
future_data['Latitude'] = future_data['Latitude'].clip(min_values['Latitude'], max_values['Latitude'])


# # Path for prediction data
CURR_DIR_PATH = os.getcwd()
PRED_DATA_PATH = CURR_DIR_PATH + "/data/clean-data/"
# Ski Centers: Salla.csv, Kommattivaara.csv, Ruskotunturi.csv, Himos.csv, Salpausselkä.csv
SKI_CENTER_PATH = PRED_DATA_PATH + "Salla.csv" # Change the .csv in quotes as needed

# # Path for trained models
MODELS_PATH = CURR_DIR_PATH + "/ml_models/"
POLY_REG_PATH = MODELS_PATH + "poly-reg/"
LINEAR_REG_PATH = MODELS_PATH + "linear-reg/"

# Read data to predict & transform date
pred_df = future_data

pred_df['Date'] = pd.to_datetime(pred_df['Date'])
pred_df['DayOfYear'] = pred_df['Date'].dt.dayofyear
pred_df['Year'] = pred_df['Date'].dt.year

# Determine X, y for prediction data 
X_test = pred_df[['Cloud_value', 'Average temperature [°C]', 'Solar Radiation (kW-hr/m^2/day)', 'Latitude', 'DayOfYear', 'Year']]

# Polynomial Regression Load
poly_features = joblib.load(POLY_REG_PATH + "model-poly-reg-all-metrics.pkl") # Change feature file (.pkl) as needed
model_poly = joblib.load(POLY_REG_PATH + "model-poly-reg-all-metrics.joblib") # Change model file (.joblib) as needed

X_test_poly = poly_features.transform(X_test)
y_pred_poly = model_poly.predict(X_test_poly)

# Creating a column for snow depth prediction
pred_df['Snow depth pred.'] = y_pred_poly

import matplotlib.pyplot as plt
plt.plot(pred_df["Date"], y_pred_poly)
plt.show()