import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# # Path for training & testing data
CURR_DIR_PATH = os.getcwd()
TRAIN_DATA_PATH = CURR_DIR_PATH + "/data/machine-learning/training-set/combined_data.csv"
TEST_DATA_PATH = CURR_DIR_PATH + "/data/clean-data/Vuokatti.csv"

# # Path for trained models (polynomial)
MODELS_PATH = CURR_DIR_PATH + "/ml_models/"
LINEAR_REG_MODELS_PATH = MODELS_PATH + "linear-reg/"


# Loading Training & Testing data
training_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

# Transform date into datetime format
training_df['Date'] = pd.to_datetime(training_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])

# Extract day of year and year as new features
training_df['DayOfYear'] = training_df['Date'].dt.dayofyear
training_df['Year'] = training_df['Date'].dt.year
test_df['DayOfYear'] = test_df['Date'].dt.dayofyear
test_df['Year'] = test_df['Date'].dt.year

# Select relevant features for Training & Testing data
X_train = training_df[['Cloud_value', 'Average temperature [°C]', 'Solar Radiation (kW-hr/m^2/day)', 'Latitude', 'DayOfYear', 'Year']]
y_train = training_df['Snow depth [cm]']

X_test = test_df[['Cloud_value', 'Average temperature [°C]', 'Solar Radiation (kW-hr/m^2/day)', 'Latitude', 'DayOfYear', 'Year']]
y_test = test_df['Snow depth [cm]']

# Linear Regression fit
model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the Linear model
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Training Mean Sq. Error:", train_rmse)
print("Testing Mean Sq. Error:", test_rmse)
print("Training R^2 Score:", train_r2)
print("Testing R^2 Score:", test_r2)




""" Saving the model and polynomial features """
# joblib.dump(model, LINEAR_REG_MODELS_PATH + "model-linear-reg-all-metrics.joblib") # Change model name in quotes




""" Test plots """
# import matplotlib.pyplot as plt

# plt.scatter(y_train, y_train_pred_linear, alpha=0.2)
# plt.show()


# # Plotting actual vs. predicted values for training set
# plt.figure(figsize=(10, 6))
# plt.scatter(y_train, y_train_pred, color='blue', label='Actual vs. Predicted (Training)')
# plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
# plt.xlabel('Actual Snow Depth')
# plt.ylabel('Predicted Snow Depth')
# plt.title('Actual vs. Predicted Snow Depth (Training Set)')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plotting actual vs. predicted values for testing set
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_test_pred, color='green', label='Actual vs. Predicted (Testing)')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
# plt.xlabel('Actual Snow Depth')
# plt.ylabel('Predicted Snow Depth')
# plt.title('Actual vs. Predicted Snow Depth (Testing Set)')
# plt.legend()
# plt.grid(True)
# plt.show()


