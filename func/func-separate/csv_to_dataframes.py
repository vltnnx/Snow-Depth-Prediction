import glob
import os
import pandas as pd

def csv_to_dataframes(data_folder_path):
    """ Reading raw data from .csv files and creating DataFrames
    combining raw data based on Ski Center & metric.
     
    Parameters: 
    data_folder_path -  requires path to folder including Ski Center
                        folders
    """
    all_dataframes = []
    ski_center_folders = glob.glob(os.path.join(data_folder_path, "raw-data", "*"))
    
    for ski_center_path in ski_center_folders:
        ski_center_name = os.path.basename(ski_center_path)
        data_folders = ["cloud-cover", "snow-depth", "solar-radiation", "temperature"]
        
        for folder_name in data_folders:
            folder_path = os.path.join(ski_center_path, folder_name)
            files = glob.glob(os.path.join(folder_path, '*.csv'))
            dfs = []
            for file in files:
                if folder_name == "solar-radiation":
                    df = pd.read_csv(file, header=9)  # Reads data starting line 10
                elif folder_name == "cloud-cover":
                    df = pd.read_csv(file)
                    df = df.drop(columns="Observation station")
                    df[["Cloud_desc", "Cloud_value"]] = df["Cloud cover [1/8]"].str.split("(", expand=True)
                    df = df.drop(columns="Cloud cover [1/8]")
                    df["Cloud_value"] = df["Cloud_value"].str.replace(")", "").str.split("/").str[0]
                    df["Cloud_value"] = df["Cloud_value"].str.strip()
                    df["Cloud_value"] = pd.to_numeric(df["Cloud_value"])
                else:
                    df = pd.read_csv(file)
                    df = df.drop(columns="Observation station")
                dfs.append(df)

            concatenated_df = pd.concat(dfs, ignore_index=True)
            concatenated_df.name = f"{ski_center_name}_{folder_name}"
            all_dataframes.append(concatenated_df)
    
    return all_dataframes



# # TEST PARAMETERS FOR FUNCTION
# CURR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# data_dir = CURR_DIR_PATH + "/data/"
# all_dataframes = csv_to_dataframes(data_dir)


# # VIEWING SINGLE DATAFRAME
# print(all_dataframes[1])


# # CHECKING ALL DATA FRAMES
# for df in all_dataframes:
#     # print(df.name)
#     print(df.head())
