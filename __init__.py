from func.data_transform import *
import os

#latitude dict
ski_centers_latitudes = {
    'Himos': 61.8250,
    'Iso-syote': 65.6667,
    'Kommattivaara': 66.5000,
    'Levi': 67.8058,
    'Ruskotunturi': 66.3333,
    'Salla': 66.8376,
    'Salpausselka': 61.0320,
    'Tahko': 62.2633,
    'Vihti-ski-center': 60.4146,
    'Vuokatti': 64.1425
}

CURR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
data_dir = CURR_DIR_PATH + "/data/"
all_dataframes = csv_to_dataframes(data_dir)

for ski_center, dfs_list in all_dataframes.items():
    df = data_merge(dfs_list) 
    modify_columns(df,ski_center,ski_centers_latitudes.get(ski_center))
    frame_to_csv(df,ski_center)
