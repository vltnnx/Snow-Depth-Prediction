import pandas as pd
import os
import glob
from prophet import Prophet
import joblib
import numpy as np

CURR_DIR_PATH = os.getcwd()
CLEAN_DATA_PATH = glob.glob(os.path.join(CURR_DIR_PATH, "data/clean-data/", "*.csv"))
PREDICTED_DATA_PATH = os.path.join(CURR_DIR_PATH, "data/predicted-data/")
PREDICTED_DATA_FILE = os.path.join(PREDICTED_DATA_PATH, "predicted_data.csv")
POLY_REG_PATH = os.path.join(CURR_DIR_PATH, "ml_models/poly-reg/")


def weather_prediction(): # Using Prophet: cloud cover, solar rad., temp.
    """Loads the clean data for each ski center and extrapolates future data,
    using Prophet module, for 30 years after the last date in datasets. Calls 
    the predict_snow_depth function to use the trained Polynomial Regression model 
    to predict future snow depth.

    Formats the data columns by using functions format_historical_columns and
    format_predicted_columns.

    Saves the prediction data to data/predicted-data/predicted_data.csv.
    
    """
    future_df = pd.DataFrame()

    for file in CLEAN_DATA_PATH:
        df_load = pd.read_csv(file)
        df_load["Date"] = pd.to_datetime(df_load["Date"])

        ski_center_df = pd.DataFrame()

        for column in ["Cloud_value", "Solar Radiation (kW-hr/m^2/day)", "Average temperature [°C]"]:
            df = pd.DataFrame()
            df["ds"] = df_load["Date"]
            df["y"] = df_load[column]

            model = Prophet()
            model.fit(df)

            years = 30
            future = model.make_future_dataframe(periods=365*years)

            forecast = model.predict(future)

            ski_center_df[column] = forecast["yhat"]

        ski_center_df["Date"] = forecast["ds"]
        ski_center_df["Date"] = pd.to_datetime(ski_center_df["Date"])
        ski_center_df = ski_center_df[ski_center_df["Date"].dt.year >= 2024]
        ski_center_df["Latitude"] = df_load["Latitude"].iloc[0]
        ski_center_df["Ski Center Name"] = df_load["Ski Center Name"].iloc[0]

        # df_load_copy = df_load.copy()
        ski_center_df = predict_snow_depth(ski_center_df)

        ski_center_df.reset_index(drop=True, inplace=True)
        df_load.reset_index(drop=True, inplace=True)

        ski_center_df = format_predicted_columns(ski_center_df)
        df_load = format_historical_columns(df_load)

        ski_center_df = pd.concat([df_load, ski_center_df], ignore_index=True)
        future_df = pd.concat([future_df, ski_center_df], ignore_index=True)

    future_df.to_csv(PREDICTED_DATA_FILE, index=False)

def format_historical_columns(df):
    """Formats the historical data columns."""
    df.rename(columns={"Cloud_value":"Cloud Cover (1-8)", "Snow depth [cm]":"Snow Depth (cm)",
                       "Average temperature [°C]":"Temperature (°C)", "Ski Center Name":"Ski Center"},
                       inplace=True)
    
    df.loc[df["Snow Depth (cm)"] < 0, "Snow Depth (cm)"] = 0
    
    return df

def format_predicted_columns(df):
    """Formats the predicted data columns."""
    df.rename(columns={"Cloud_value":"Predicted Cloud Cover (1-8)", "Snow depth [cm]":"Predicted Snow Depth (cm)",
                       "Average temperature [°C]":"Predicted Temperature (°C)", "Ski Center Name":"Ski Center",
                       "Solar Radiation (kW-hr/m^2/day)":"Predicted Solar Radiation (kW-hr/m^2/day)"},
                       inplace=True)
    
    return df
    
def predict_snow_depth(ski_center_df):
    """Uses a trained Polynomial Regression model to predict snow depth
    in the future based on extrapolated future data.
    
    Parameters:
    ski_center_df:  DataFrame of extrapolated data for cloud cover, temperature
                    and solar radiation.
    """
    ski_center_df["DayOfYear"] = ski_center_df["Date"].dt.dayofyear
    ski_center_df["Year"] = ski_center_df["Date"].dt.year

    X = ski_center_df[['Cloud_value', 'Average temperature [°C]', 'Solar Radiation (kW-hr/m^2/day)', 'Latitude', 'DayOfYear', 'Year']]

    poly_features = joblib.load(POLY_REG_PATH + "model-poly-reg-all-metrics.pkl") 
    model_poly = joblib.load(POLY_REG_PATH + "model-poly-reg-all-metrics.joblib")

    X_poly = poly_features.transform(X)
    y_pred = model_poly.predict(X_poly)
    y_pred_non_neg = np.maximum(y_pred, 0)

    ski_center_df["Predicted Snow Depth (cm)"] = y_pred_non_neg

    ski_center_df = ski_center_df.drop(["DayOfYear", "Year"], axis=1)

    return ski_center_df



weather_prediction()