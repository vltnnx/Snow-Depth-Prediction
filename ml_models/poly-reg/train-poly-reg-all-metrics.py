import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# # Path for training & testing data
CURR_DIR_PATH = os.getcwd()
TRAIN_DATA_PATH = CURR_DIR_PATH + "/data/machine-learning/training-set/combined_data.csv"
TEST_DATA_PATH = CURR_DIR_PATH + "/data/clean-data/Vuokatti.csv"

# # Path for trained models (polynomial)
MODELS_PATH = CURR_DIR_PATH + "/ml_models/"
POLY_REG_MODELS_PATH = MODELS_PATH + "poly-reg/"

# Load training & testing data
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

# Transform date into datetime format, extract day of year and year as new features
train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])

train_df['DayOfYear'] = train_df['Date'].dt.dayofyear
train_df['Year'] = train_df['Date'].dt.year
test_df['DayOfYear'] = test_df['Date'].dt.dayofyear
test_df['Year'] = test_df['Date'].dt.year

# Select features for training & testing data X & y
X_train = train_df[['Cloud_value', 'Average temperature [°C]', 'Solar Radiation (kW-hr/m^2/day)', 'Latitude', 'DayOfYear', 'Year']]
y_train = train_df['Snow depth [cm]']

X_test = test_df[['Cloud_value', 'Average temperature [°C]', 'Solar Radiation (kW-hr/m^2/day)', 'Latitude', 'DayOfYear', 'Year']]
y_test = test_df['Snow depth [cm]']

# Generate polynomial features & Train polynomial reg. model
degree = 3  # Choose the degree of the polynomial
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predictions: Training data, Testing data & Testing data with non-negative values 
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)
y_test_pred_non_negative = np.maximum(y_test_pred, 0)


""" Saving the model and polynomial features """
# joblib.dump(model, POLY_REG_MODELS_PATH + "model-poly-reg-all-metrics.joblib") # Change model name in quotes
# joblib.dump(poly_features, POLY_REG_MODELS_PATH + "model-poly-reg-all-metrics.pkl") # Change feature name in quotes


""" Evaluate the model """
# train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
# test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
# train_r2 = r2_score(y_train, y_train_pred)
# test_r2 = r2_score(y_test, y_test_pred)

# print("Training Mean Sq. Error:", train_rmse)
# print("Testing Mean Sq. Error:", test_rmse)
# print("Training R^2 Score:", train_r2)
# print("Testing R^2 Score:", test_r2)


""" Evaluate the model non-negative test """
# train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
# test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_non_negative))
# train_r2 = r2_score(y_train, y_train_pred)
# test_r2 = r2_score(y_test, y_test_pred_non_negative)

# print("Training RMSE:", train_rmse)
# print("Testing RMSE:", test_rmse)
# print("Training R^2 Score:", train_r2)
# print("Testing R^2 Score:", test_r2)


""" Visualizing prediction VS. True"""
# import matplotlib.pyplot as plt

# # Fit a linear regression model to the predicted values
# regression_model = LinearRegression()
# regression_model.fit(np.arange(len(y_test)).reshape(-1, 1), y_test_pred_non_negative)

# # Plot the trendline
# plt.plot(test_df["Date"], regression_model.predict(np.arange(len(y_test)).reshape(-1, 1)), linestyle='--', label='Trendline', color='green')

# # Plot Predicted Snow and Actual Snow
# plt.plot(test_df["Date"], y_test_pred_non_negative, label="Predicted Snow Depth")
# plt.plot(test_df["Date"], y_test, color="r", label="Actual Snow Depth")

# plt.legend()
# plt.xlabel('Date')
# plt.ylabel('Snow Depth (cm)')
# plt.title('Actual vs Predicted Snow Depth')

# plt.show()


# """ Saving the model and polynomial features """
# joblib.dump(model, "snow-depth-polynomial-regression-all-metrics.joblib")
# joblib.dump(poly_features, "snow-depth-polynomial-regression-all-metrics-features.pkl")


""" Test plots """
# import matplotlib.pyplot as plt

# plt.scatter(y_test, y_test_pred_non_negative, alpha=0.2)
# plt.scatter(y_test, y_test_pred, color="r", alpha=0.2)
# plt.show()

# plt.scatter(y_train, y_train_pred, alpha=0.2)
# plt.plot()
# plt.show()

""" Saving the model and polynomial features """
# joblib.dump(model, "snow-depth-polynomial-regression-all-metrics.joblib")
# joblib.dump(poly_features, "snow-depth-polynomial-regression-all-metrics-features.pkl")