import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Ski Centers: Salla, Kommattivaara, Ruskotunturi, Himos, Salpausselkä
SKI_CENTER = "Salpausselka"

# # Path for prediction data
CURR_DIR_PATH = os.getcwd()
PRED_DATA_PATH = CURR_DIR_PATH + "/data/clean-data/"
SKI_CENTER_PATH = PRED_DATA_PATH + SKI_CENTER + ".csv"

# # Path for trained models
MODELS_PATH = CURR_DIR_PATH + "/ml_models/"
POLY_REG_PATH = MODELS_PATH + "poly-reg/"
LINEAR_REG_PATH = MODELS_PATH + "linear-reg/"

# Read historical data & transform date
pred_df = pd.read_csv(SKI_CENTER_PATH)

pred_df['Date'] = pd.to_datetime(pred_df['Date'])
pred_df['DayOfYear'] = pred_df['Date'].dt.dayofyear
pred_df['Year'] = pred_df['Date'].dt.year

# Determine X, y for prediction data 
predict = pred_df[['Cloud_value', 'Average temperature [°C]', 'Solar Radiation (kW-hr/m^2/day)', 'Latitude', 'DayOfYear', 'Year']]
# y_test = pred_df['Snow depth [cm]'] # Not needed for prediction, can use when comparing with prediction

# Polynomial Regression Load
poly_features = joblib.load(POLY_REG_PATH + "model-poly-reg-all-metrics.pkl") # Change feature file (.pkl) as needed
model_poly = joblib.load(POLY_REG_PATH + "model-poly-reg-all-metrics.joblib") # Change model file (.joblib) as needed

X_test_poly = poly_features.transform(predict)
y_pred_poly = model_poly.predict(X_test_poly)
y_pred_non_negative = np.maximum(y_pred_poly, 0)

# # Creating a column for snow depth prediction
pred_df["Snow depth pred. non neg"] = y_pred_non_negative

# Define a function to plot trendlines
def plot_trendline(ax, x, y, color, label):
    # Convert dates to numeric values
    x_num = mdates.date2num(x)
    
    # Fit a linear trendline
    coef = np.polyfit(x_num, y, 1)
    poly = np.poly1d(coef)
    
    # Plot the trendline
    ax.plot(x, poly(x_num), color=color, linestyle='--', label=label)

# Create a new figure and set its size
plt.figure(figsize=(10, 6))

# Plot actual snow depth
plt.plot(pred_df["Date"], pred_df["Snow depth [cm]"], label='Actual Snow Depth')

# Plot predicted snow depth
plt.plot(pred_df["Date"], pred_df['Snow depth pred. non neg'], color="r", alpha=0.2, label='Predicted Snow Depth')

# Add trendlines
plot_trendline(plt.gca(), pred_df["Date"], pred_df["Snow depth [cm]"], color='blue', label='Actual Trend')
plot_trendline(plt.gca(), pred_df["Date"], pred_df['Snow depth pred. non neg'], color='red', label='Predicted Trend')

# Set plot title and labels
plt.title("Salpausselkä - Snow Depth")
plt.ylabel("Snow Depth (cm)")
plt.legend()  # Show legend with labels

plt.show()













""" Save to csv """
# future_df.to_csv("C:\\Users\\Joonas\\Desktop\\snow-predict.csv")


""" Evaluate the model """
# from sklearn.metrics import mean_squared_error, r2_score

# # Calculate RMSE
# rmse = mean_squared_error(pred_df['Snow depth [cm]'], pred_df['Snow depth pred.'], squared=False)
# print("RMSE:", rmse)
# # Calculate R-squared
# r_squared = r2_score(pred_df['Snow depth [cm]'], pred_df['Snow depth pred.'])
# print("R-squared:", r_squared)




# import matplotlib.pyplot as plt

# plt.plot(pred_df["Date"], pred_df["Snow depth [cm]"], label='Actual Snow Depth')
# plt.plot(pred_df["Date"], pred_df['Snow depth pred. non neg'], color="r", alpha=0.2, label='Predicted Snow Depth')
# plt.title("Salpausselkä - Snow Depth")
# plt.ylabel("Snow Depth (cm)")
# plt.legend()  # Show legend with labels

# plt.show()
