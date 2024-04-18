import pandas as pd

def data_merge(frame_list):
    '''Takes a list of 4 dataframes and merges them to one. Use Year, Month, Day as columns to join on'''
    new_frame = frame_list[0].merge(frame_list[1], on=['Year', 'Month', 'Day'])
    new_frame = new_frame.merge(frame_list[2], on=['Year', 'Month', 'Day'])
    new_frame = new_frame.merge(frame_list[3], on=['Year', 'Month', 'Day'])
    return new_frame

def frame_to_csv(frame, name):
    '''Takes a dataframe and a name, and creates a csv-file to path Data/clean-data'''
    filepath = 'Data/clean-data/' + name + '.csv'
    frame.to_csv(filepath)



