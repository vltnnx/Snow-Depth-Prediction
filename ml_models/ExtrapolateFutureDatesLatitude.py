import pandas as pd

def extrapolate_date_latitude(ski_center):
    # Dictionary of ski centers and their latitudes
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

    # Specify the ski center
    selected_ski_center = ski_centers_latitudes[ski_center]

    # Generate date range from 2024-01-01 to 2050-12-31
    start_date = '2024-01-01'
    end_date = '2050-12-31'
    date_range = pd.date_range(start=start_date, end=end_date)

    # Create DataFrame with the date range
    df = pd.DataFrame({'Date': date_range})

    # Extract day of the year and year into new columns
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Year'] = df['Date'].dt.year

    # Add a column for the latitude of the selected ski center
    df['Latitude'] = selected_ski_center

    # Display the DataFrame
    return df


# extrapolate_date_latitude("Himos")