import glob
import os
import pandas as pd

def transform_hourly_avgs(dataframes):
    """Transforms the hourly measurements into a uniform daily data format."""
    dataframes.set_index(['Year', 'Month', 'Day'], inplace=True)
    df_daily_avg = dataframes.groupby(level=['Year', 'Month', 'Day'])[dataframes.columns[-1]].mean()
    dataframes[dataframes.columns[-1]] = df_daily_avg
    dataframes.drop(dataframes.columns[-2], axis=1, inplace=True)
    dataframes.reset_index(inplace=True)  # reset index to access 'Year', 'Month', and 'Day' as columns
    dataframes.drop_duplicates(subset=['Year', 'Month', 'Day'], inplace=True)

def csv_to_dataframes(data_folder_path):
    """ Reads raw data from .csv files, returns a dictionary:
    - keys = ski center name
    - values = DataFrame for each metric in a list

    1. Reads ski center names from folders in raw-data folder
    2. Loops through each metric folder for each ski center
       to create the DataFrames and modifies the data based on
       the needs recognised from each data set for metrics.
    3. Uses transform_hourly_avgs function to calculate daily
       averages, replacing hourly measurements in the DataFrames
       where hourly measurements were used.
    4. Uses clean_data function to drop rows with no measurement
       data (null in all, -999 in solar radiation data set).
     
    Parameters: 
    data_folder_path -  path to the folder including ski center
                        folders
    """
    all_dataframes = {}
    ski_center_folders = glob.glob(os.path.join(data_folder_path, "raw-data", "*"))
    
    for ski_center_path in ski_center_folders:
        ski_center_name = os.path.basename(ski_center_path)
        data_folders = ["cloud-cover", "snow-depth", "solar-radiation", "temperature"]
        ski_center_dataframes = []
        
        for folder_name in data_folders:
            folder_path = os.path.join(ski_center_path, folder_name)
            files = glob.glob(os.path.join(folder_path, '*.csv'))
            dfs = []
            for file in files:
                if folder_name == "solar-radiation":
                    df = pd.read_csv(file, header=9)  # Reads data starting line 10
                elif folder_name == "cloud-cover":
                    df = pd.read_csv(file)
                    df = df.drop(df.columns[0], axis=1)
                    df[["Cloud_desc", "Cloud_value"]] = df["Cloud cover [1/8]"].str.split("(", expand=True)
                    df = df.drop(columns="Cloud cover [1/8]")
                    df = df.drop(columns="Cloud_desc")
                    df["Cloud_value"] = df["Cloud_value"].str.replace(")", "").str.split("/").str[0]
                    df["Cloud_value"] = df["Cloud_value"].str.strip()
                    df["Cloud_value"] = pd.to_numeric(df["Cloud_value"])
                else:
                    df = pd.read_csv(file)
                    df = df.drop(df.columns[0], axis=1)
                    df.iloc[:, -1] = pd.to_numeric(df.iloc[:, -1], errors='coerce')
                dfs.append(df)

            concatenated_df = pd.concat(dfs, ignore_index=True)
            # Rename columns if they are present, mainly for solar data
            if all(col in concatenated_df.columns for col in ["YEAR", "MO", "DY"]):
                concatenated_df.rename(columns={'YEAR': 'Year', 'MO': 'Month', 'DY': 'Day'}, inplace=True)

            if concatenated_df.duplicated(subset=('Year','Month','Day')).any():
                transform_hourly_avgs(concatenated_df)
            clean_data(concatenated_df)    
            concatenated_df.name = f"{ski_center_name}"
            ski_center_dataframes.append(concatenated_df)
 
        all_dataframes[ski_center_name] = ski_center_dataframes

    return all_dataframes

def clean_data(dataframe):
    """Removes null values from DataFrames"""
    # Drops rows if null values
    dataframe.dropna(how='any', axis=0, inplace=True)

    # Drop radiation rows where value is -999.0
    if all(col in dataframe.columns for col in ["ALLSKY_SFC_SW_DWN"]):
        index_999 = dataframe[dataframe["ALLSKY_SFC_SW_DWN"] == -999.0].index
        dataframe.drop(index_999, inplace=True)

def data_merge(frame_list):
    """Merges the DataFrames for each measurement into a single
    DataFrame for each ski center on "Year", "Month", "Day"
    columns for the join.

    Parameters:
    - frame_list:   list of four (4) DataFrames
    """
    new_frame = frame_list[0].merge(frame_list[1], on=['Year', 'Month', 'Day'])
    new_frame = new_frame.merge(frame_list[2], on=['Year', 'Month', 'Day'])
    new_frame = new_frame.merge(frame_list[3], on=['Year', 'Month', 'Day'])

    return new_frame

def modify_columns(dataframe, skicenter_name, latitude):
    """Modifies the columns in DataFrames

    1. Creates a single "Date" column from "Year", "Month", "Day" columns
    2. Adds "Ski Center Name" and "Latitude" columns in the DataFrames
    3. Renames solar radiation column
    4. Drops unnecessary columns
    """

    dataframe['Date'] = pd.to_datetime(dataframe[['Year', 'Month', 'Day']])
    
    # Add 'Ski Center Name' and 'Latitude' columns
    dataframe['Ski Center Name'] = skicenter_name
    dataframe['Latitude'] = latitude
    # Drop columns 'Year', 'Month', 'Day'
    dataframe.drop(columns=['Year', 'Month', 'Day', 'Time [Local time]_x', 'Time [Local time]_y'], inplace=True)
    if all(col in dataframe.columns for col in ['Time [Local time]']):
        dataframe.drop(columns='Time [Local time]', inplace=True)
    dataframe.rename(columns={"ALLSKY_SFC_SW_DWN": "Solar Radiation (kW-hr/m^2/day)"}, inplace=True)

def frame_to_csv(frame, name):
    """Creates a .csv file from a cleaned ski center DataFrame
    into data/clean-data folder.

    Parameters:
    - frame: ski center DataFrame
    - name: ski center name
    """
    filepath = 'Data/clean-data/' + name + '.csv'
    frame.to_csv(filepath, index=False)
