import pandas as pd

def clean_data(dataframe):
    """"Remove null values from dataframe and standardize names"""
    
    # Make a copy of the original DataFrame
    dataframe.dropna(how='any', axis=0, inplace=True)
    
    # Rename columns if they are present, mainly for solar data
    if all(col in dataframe.columns for col in ["YEAR", "MO", "DY"]):
        dataframe.rename(columns={'YEAR': 'Year', 'MO': 'Month', 'DY': 'Day'}, inplace=True)
