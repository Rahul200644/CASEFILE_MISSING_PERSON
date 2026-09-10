import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import joblib

def perform_kmeans_clustering(df, n_clusters=5, lat_col='latitude', lon_col='longitude'):
    """
    Cluster GPS coordinates into frequent areas using K-Means.
    """
    coords = df[[lat_col, lon_col]].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(coords)
    
    df_clustered = df.copy()
    df_clustered['cluster_kmeans'] = cluster_labels
    
    centroids = kmeans.cluster_centers_
    cluster_info = {}
    area_names = ['Area A', 'Area B', 'Area C', 'Area D', 'Area E', 'Area F', 'Area G']
    
    for c in range(n_clusters):
        count = (cluster_labels == c).sum()
        cluster_info[c] = {
            'area_name': area_names[c % len(area_names)],
            'centroid_lat': float(centroids[c][0]),
            'centroid_lon': float(centroids[c][1]),
            'visit_count': int(count),
            'visit_ratio': float(count / len(df))
        }
        
    df_clustered['area_name'] = df_clustered['cluster_kmeans'].map(lambda x: cluster_info[x]['area_name'])
    return df_clustered, kmeans, cluster_info

def perform_dbscan_clustering(df, eps_km=0.5, min_samples=10, lat_col='latitude', lon_col='longitude'):
    """
    Cluster GPS stay points using DBSCAN with Haversine metric.
    """
    coords_rad = np.radians(df[[lat_col, lon_col]].values)
    kms_per_radian = 6371.0088
    epsilon = eps_km / kms_per_radian
    
    db = DBSCAN(eps=epsilon, min_samples=min_samples, metric='haversine')
    cluster_labels = db.fit_predict(coords_rad)
    
    df_clustered = df.copy()
    df_clustered['cluster_dbscan'] = cluster_labels
    return df_clustered, db
