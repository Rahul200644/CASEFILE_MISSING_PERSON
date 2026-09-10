import numpy as np
import pandas as pd

def validate_coordinates(df, lat_col='latitude', lon_col='longitude'):
    """
    Remove invalid GPS coordinates outside valid geographical bounds.
    """
    valid_lat = (df[lat_col] >= -90.0) & (df[lat_col] <= 90.0)
    valid_lon = (df[lon_col] >= -180.0) & (df[lon_col] <= 180.0)
    return df[valid_lat & valid_lon].copy()

def remove_duplicates(df, subset=None):
    """
    Remove duplicate GPS trajectory records.
    """
    return df.drop_duplicates(subset=subset).copy()

def handle_missing_values(df):
    """
    Clean and handle missing values in numeric and categorical columns.
    """
    df = df.dropna(subset=['latitude', 'longitude']).copy()
    if 'speed' in df.columns:
        df['speed'] = df['speed'].fillna(0.0)
    if 'distance' in df.columns:
        df['distance'] = df['distance'].fillna(0.0)
    return df

def extract_time_features(df, timestamp_col='datetime'):
    """
    Convert timestamp into hour, day, weekday, month, and weekend indicator.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    df['hour'] = df[timestamp_col].dt.hour
    df['day'] = df[timestamp_col].dt.day
    df['day_of_week'] = df[timestamp_col].dt.day_name()
    df['weekday'] = df[timestamp_col].dt.weekday
    df['month'] = df[timestamp_col].dt.month
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    
    return df

def preprocess_gps_data(df, timestamp_col='datetime'):
    """
    Full preprocessing pipeline for GPS trajectory datasets.
    """
    df = validate_coordinates(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = extract_time_features(df, timestamp_col=timestamp_col)
    return df
