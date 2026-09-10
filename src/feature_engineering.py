import numpy as np
import pandas as pd

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate Haversine distance in kilometers between two GPS points.
    """
    R = 6371.0 # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def compute_movement_metrics(df, user_col='user_id', lat_col='latitude', lon_col='longitude', time_col='datetime'):
    """
    Derive distance between consecutive points, movement duration, and speed.
    """
    df = df.sort_values(by=[user_col, time_col]).copy()
    
    df['prev_lat'] = df.groupby(user_col)[lat_col].shift(1)
    df['prev_lon'] = df.groupby(user_col)[lon_col].shift(1)
    df['prev_time'] = df.groupby(user_col)[time_col].shift(1)
    
    # Distance in km
    df['step_distance_km'] = calculate_haversine_distance(
        df['prev_lat'], df['prev_lon'], df[lat_col], df[lon_col]
    ).fillna(0.0)
    
    # Duration in seconds
    df['duration_sec'] = (df[time_col] - df['prev_time']).dt.total_seconds().fillna(0.0)
    
    # Speed in km/h
    duration_hours = df['duration_sec'] / 3600.0
    df['calculated_speed_kmh'] = np.where(
        duration_hours > 0, df['step_distance_km'] / duration_hours, 0.0
    )
    df['calculated_speed_kmh'] = df['calculated_speed_kmh'].clip(upper=150.0) # Cap unrealistic spikes
    
    df = df.drop(columns=['prev_lat', 'prev_lon', 'prev_time'])
    return df

def aggregate_user_movement_profiles(df, user_col='user_id'):
    """
    Aggregate movement statistics per user/person profile.
    """
    profile = df.groupby(user_col).agg(
        total_records=('latitude', 'count'),
        total_distance_km=('step_distance_km', 'sum'),
        avg_speed_kmh=('calculated_speed_kmh', 'mean'),
        max_speed_kmh=('calculated_speed_kmh', 'max'),
        start_time=('datetime', 'min'),
        end_time=('datetime', 'max')
    ).reset_index()
    return profile
