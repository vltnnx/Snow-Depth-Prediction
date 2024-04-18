import pandas as pd


def transform_hourly_avgs(dataframes):
    """Transform the hourly data into a uniform daily data format"""
    for attribute, df in dataframes.items(): # might need to be changed
        df.set_index(['Year', 'Month', 'Day'], inplace=True)
        df_daily_avg = df.groupby(level=['Year', 'Month', 'Day'])[df.columns[-1]].mean()
        df[df.columns[-1]] = df_daily_avg # overwrites the original file
        df.drop(df.columns[-2], axis=1, inplace=True)
        df.reset_index(inplace=True)  # reset index to access 'Year', 'Month', and 'Day' as columns
        df.drop_duplicates(subset=['Year', 'Month', 'Day'], inplace=True)
