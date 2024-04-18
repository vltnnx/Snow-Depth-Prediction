import pandas as pd

def data_transform(dataframe, skicenter_name, latitude):
    """Transform the data by adding date, ski center name, and latitude columns."""

    dataframe['Date'] = pd.to_datetime(dataframe[['Year', 'Month', 'Day']])
    
    # Add 'Ski Center Name' and 'Latitude' columns
    dataframe['Ski Center Name'] = skicenter_name
    dataframe['Latitude'] = latitude
    # Drop columns 'Year', 'Month', 'Day'
    dataframe.drop(columns=['Year', 'Month', 'Day'], inplace=True)
    if all(col in dataframe.columns for col in ['Time [Local time]']):
        dataframe.drop(columns='Time [Local time]', inplace=True)
