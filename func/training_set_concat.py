import os
import pandas as pd

CURR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
folder_path = os.path.join(CURR_DIR_PATH + '\\data\\clean-data\\')
training_set_path = os.path.join(CURR_DIR_PATH + '\\data\\machine-learning\\training-set\\combined_data.csv')

# Read each CSV file into a DataFrame: Levi, Vihti, Tahko, Iso-syote
df1 = pd.read_csv(os.path.join(folder_path,'Levi.csv'))
df2 = pd.read_csv(os.path.join(folder_path,'Vihti-ski-center.csv'))
df3 = pd.read_csv(os.path.join(folder_path,'Tahko.csv'))
df4 = pd.read_csv(os.path.join(folder_path,'Iso-syote.csv'))

# Concatenate the DataFrames into a single DataFrame
training_set_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Save the training_set_df to machine learning folder
training_set_df.to_csv(training_set_path, index=False)
