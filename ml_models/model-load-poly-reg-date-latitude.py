import pandas as pd
import joblib
import os
from ExtrapolateFutureDatesLatitude import *

# # Path for prediction data
CURR_DIR_PATH = os.getcwd()
PRED_DATA_PATH = CURR_DIR_PATH + "/data/clean-data/"
# Ski Centers: Salla.csv, Kommattivaara.csv, Ruskotunturi.csv, Himos.csv, Salpausselkä.csv
SKI_CENTER_PATH = PRED_DATA_PATH + "Kommattivaara.csv" # Change the .csv in quotes as needed

# # Path for trained models
MODELS_PATH = CURR_DIR_PATH + "/ml_models/"
POLY_REG_PATH = MODELS_PATH + "poly-reg/"

# Extrapolate future data (dayofyear, year, latitude) for ski center
pred_df = extrapolate_date_latitude("Kommattivaara")

# pred_df['Date'] = pd.to_datetime(pred_df['Date'])
# pred_df['DayOfYear'] = pred_df['Date'].dt.dayofyear
# pred_df['Year'] = pred_df['Date'].dt.year

# Determine X, y for prediction data 
X_test = pred_df[['Latitude', 'DayOfYear', 'Year']]
# y_test = pred_df['Snow depth [cm]'] # Not needed for prediction, can use when comparing with prediction

# Polynomial Regression Load
poly_features = joblib.load(POLY_REG_PATH + "model-poly-reg-day-year-latitude.pkl") # Change feature file (.pkl) as needed
model_poly = joblib.load(POLY_REG_PATH + "model-poly-reg-day-year-latitude.joblib") # Change model file (.joblib) as needed

X_test_poly = poly_features.transform(X_test)
y_pred_poly = model_poly.predict(X_test_poly)

# Creating a column for snow depth prediction
pred_df['Snow depth pred.'] = y_pred_poly



""" Plot snow depth actual VS. predicted"""
# import matplotlib.pyplot as plt

# salla_df = pd.read_csv(SKI_CENTER_PATH)

# salla_df['Date'] = pd.to_datetime(salla_df['Date'])

# salla_plot = salla_df[["Date", "Snow depth [cm]"]]


# plt.plot(pred_df['Date'], pred_df['Snow depth pred.'])
# plt.plot(salla_plot['Date'], salla_plot["Snow depth [cm]"])
# plt.show()

