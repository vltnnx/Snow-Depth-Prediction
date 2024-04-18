import pandas as pd
import os
from prophet import Prophet

CURR_DIR_PATH = os.getcwd()
PRED_DATA_PATH = CURR_DIR_PATH + "/data/clean-data/"

def extrapolate_data(ski_center):
    """Using Prophet, xtrapolates future data for cloud cover, 
    solar radiation, and average temperature based on historical data
    
    Parameters:
    - ski_center:   Name of a ski center to read historical data
                    from a .csv
    """
    ski_center_path = PRED_DATA_PATH + ski_center + ".csv"

    # Read data to predict & transform date
    df_load = pd.read_csv(ski_center_path)
    df_load['Date'] = pd.to_datetime(df_load['Date'])

    # Initialize an empty DataFrame to store predictions
    predictions_df = pd.DataFrame()

    # Iterate over each column to predict
    for column in ['Cloud_value', 'Solar Radiation (kW-hr/m^2/day)', 'Average temperature [°C]']:
        # Create a DataFrame with 'ds' and the current column as 'y'
        df = pd.DataFrame()
        df['ds'] = df_load['Date']
        df['y'] = df_load[column]  # Select the current column as 'y'

        # Initialize Prophet model
        m = Prophet()

        # Fit the model
        m.fit(df)

        # Create future DataFrame for forecasting
        years = 26
        future = m.make_future_dataframe(periods=366*years)

        # Predict
        forecast = m.predict(future)

        # Add the predicted column to the predictions DataFrame with the original column name
        predictions_df[column] = forecast['yhat']
        predictions_df[column + "_y_lo"] = forecast['yhat_lower']
        predictions_df[column + "_y_hi"] = forecast['yhat_upper']

    # Add the 'ds' column to the predictions DataFrame
    predictions_df['Date'] = forecast['ds']
    predictions_df['Latitude'] = df_load['Latitude'].iloc[0]
    predictions_df = predictions_df[predictions_df["Date"].dt.year >= 2024]

    return predictions_df


""" Test the function to view extrapolated data in a Matplotlib figure """
# predict = extrapolate_data("Salla")

# import matplotlib.pyplot as plt

# plt.plot(predict["Date"], predict["Average temperature [°C]"])
# plt.show()