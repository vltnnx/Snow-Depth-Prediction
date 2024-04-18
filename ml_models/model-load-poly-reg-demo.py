import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Detemine Ski Center to Predict & Extrapolate: Salla, Kommattivaara, Ruskotunturi, Himos, Salpausselkä
SKI_CENTER = "Salpausselka"

# # Path for prediction data
CURR_DIR_PATH = os.getcwd()
PRED_DATA_PATH = CURR_DIR_PATH + "/data/clean-data/"
SKI_CENTER_PATH = PRED_DATA_PATH + SKI_CENTER + ".csv"

# Path for trained models
MODELS_PATH = CURR_DIR_PATH + "/ml_models/"
POLY_REG_PATH = MODELS_PATH + "poly-reg/"
LINEAR_REG_PATH = MODELS_PATH + "linear-reg/"

import sys
sys.path.append(CURR_DIR_PATH)
from prophetFunctions.extrapolate_Prophet import extrapolate_data

# Read historical data & transform date
pred_df = pd.read_csv(SKI_CENTER_PATH)

pred_df['Date'] = pd.to_datetime(pred_df['Date'])
pred_df['DayOfYear'] = pred_df['Date'].dt.dayofyear
pred_df['Year'] = pred_df['Date'].dt.year

# Extrapolate future data with Prophet
future_df = extrapolate_data(SKI_CENTER)

future_df['Date'] = pd.to_datetime(future_df['Date'])
future_df['DayOfYear'] = future_df['Date'].dt.dayofyear
future_df['Year'] = future_df['Date'].dt.year


# Determine data to predict with 
X_future = future_df[['Cloud_value', 'Average temperature [°C]', 'Solar Radiation (kW-hr/m^2/day)', 'Latitude', 'DayOfYear', 'Year']]

# Polynomial Regression Load
poly_features = joblib.load(POLY_REG_PATH + "model-poly-reg-all-metrics.pkl") # Change feature file (.pkl) as needed
model_poly = joblib.load(POLY_REG_PATH + "model-poly-reg-all-metrics.joblib") # Change model file (.joblib) as needed

X_test_poly = poly_features.transform(X_future)
y_pred_poly = model_poly.predict(X_test_poly)
y_pred_non_negative = np.maximum(y_pred_poly, 0)

# # Creating a column for snow depth prediction
future_df["Snow depth pred. non neg"] = y_pred_non_negative




""" Plot historical and future data with their trendlines """

# Define a function to plot trendlines
def plot_trendline(ax, x, y, color, label):
    # Convert dates to numeric values
    x_num = mdates.date2num(x)
    
    # Fit a polynomial trendline
    degree = 1  # Adjust the degree of the polynomial as needed
    coef = np.polyfit(x_num, y, degree)
    poly = np.poly1d(coef)
    
    # Plot the trendline
    ax.plot(x, poly(x_num), color=color, linestyle='--', label=label)

# Figure & Title
fig = plt.figure(figsize=(10, 8))

fig.suptitle(SKI_CENTER, fontsize=16)

# Subplots
plt.subplot(2, 2, 1)  # Top-left subplot
plt.plot(future_df['Date'], future_df['Average temperature [°C]'])
plt.plot(pred_df['Date'], pred_df['Average temperature [°C]'], color="r")
plot_trendline(plt.gca(), future_df['Date'], future_df['Average temperature [°C]'], color='gray', label='Future Trend')
plot_trendline(plt.gca(), pred_df['Date'], pred_df['Average temperature [°C]'], color='blue', label='Historical Trend')
plt.title('Average Temperature')
plt.legend()

plt.subplot(2, 2, 2)  # Top-right subplot
plt.plot(future_df['Date'], future_df['Cloud_value'])
plt.plot(pred_df['Date'], pred_df['Cloud_value'], color="r")
plot_trendline(plt.gca(), future_df['Date'], future_df['Cloud_value'], color='gray', label='Future Trend')
plot_trendline(plt.gca(), pred_df['Date'], pred_df['Cloud_value'], color='blue', label='Historical Trend')
plt.title('Cloud cover')
plt.legend()

plt.subplot(2, 2, 3)  # Bottom-left subplot
plt.plot(future_df['Date'], future_df['Solar Radiation (kW-hr/m^2/day)'])
plt.plot(pred_df['Date'], pred_df['Solar Radiation (kW-hr/m^2/day)'], color="r")
plot_trendline(plt.gca(), future_df['Date'], future_df['Solar Radiation (kW-hr/m^2/day)'], color='gray', label='Future Trend')
plot_trendline(plt.gca(), pred_df['Date'], pred_df['Solar Radiation (kW-hr/m^2/day)'], color='blue', label='Historical Trend')
plt.title('Solar radiation')
plt.legend()

plt.subplot(2, 2, 4)  # Bottom-right subplot
plt.plot(future_df['Date'], future_df["Snow depth pred. non neg"])
plt.plot(pred_df['Date'], pred_df["Snow depth [cm]"], color="r")
plot_trendline(plt.gca(), future_df['Date'], future_df["Snow depth pred. non neg"], color='gray', label='Future Trend')
plot_trendline(plt.gca(), pred_df['Date'], pred_df["Snow depth [cm]"], color='blue', label='Historical Trend')
plt.title('Snow depth')
plt.legend()


plt.tight_layout()
plt.show()

