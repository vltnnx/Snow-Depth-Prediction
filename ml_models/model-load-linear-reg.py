import pandas as pd
import joblib
import os

# # Path for prediction data
CURR_DIR_PATH = os.getcwd()
PRED_DATA_PATH = CURR_DIR_PATH + "/data/clean-data/"
# Ski Centers: Salla.csv, Kommattivaara.csv, Ruskotunturi.csv, Himos.csv, Salpausselkä.csv
SKI_CENTER_PATH = PRED_DATA_PATH + "Salla.csv"

# # Path for trained models
LINEAR_REG_MODELS_PATH = CURR_DIR_PATH + "/ml_models/linear-reg/"
MODEL_PATH = LINEAR_REG_MODELS_PATH + "model-linear-reg-all-metrics.joblib" # Change model name in quotes as necessary

# Read data to predict
pred_df = pd.read_csv(SKI_CENTER_PATH)

# Transform date
pred_df['Date'] = pd.to_datetime(pred_df['Date'])
pred_df['DayOfYear'] = pred_df['Date'].dt.dayofyear
pred_df['Year'] = pred_df['Date'].dt.year

# Determine X, y from prediction data
X_test = pred_df[['Cloud_value', 'Average temperature [°C]', 'Solar Radiation (kW-hr/m^2/day)', 'Latitude', 'DayOfYear', 'Year']]
y_test = pred_df['Snow depth [cm]']

# Linear Regression Load
model = joblib.load(MODEL_PATH)

# Prediction
y_pred = model.predict(X_test)

# Creating a column for snow depth prediction
pred_df['Snow depth pred.'] = y_pred


""" Evaluate the model """
from sklearn.metrics import mean_squared_error, r2_score

# Calculate RMSE
rmse = mean_squared_error(pred_df['Snow depth [cm]'], pred_df['Snow depth pred.'], squared=False)
print("RMSE:", rmse)
# Calculate R-squared
r_squared = r2_score(pred_df['Snow depth [cm]'], pred_df['Snow depth pred.'])
print("R-squared:", r_squared)




""" Test plot """
# import matplotlib.pyplot as plt

# plt.plot(pred_df["Date"], y_pred)
# plt.show()

