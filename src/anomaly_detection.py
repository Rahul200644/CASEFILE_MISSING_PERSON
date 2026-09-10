import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import joblib

def train_isolation_forest(X, contamination=0.05, random_state=42):
    """
    Train Isolation Forest for movement anomaly detection.
    """
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso.fit(X)
    return iso

def train_local_outlier_factor(X, n_neighbors=20, contamination=0.05):
    """
    Train Local Outlier Factor model.
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)
    lof.fit(X)
    return lof

def train_one_class_svm(X, nu=0.05, kernel='rbf', gamma='scale'):
    """
    Train One-Class SVM model.
    """
    ocsvm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    ocsvm.fit(X)
    return ocsvm

def detect_anomalies(df, model, feature_cols):
    """
    Predict anomaly flags (-1: anomaly, 1: normal) and decision scores.
    """
    X = df[feature_cols].copy()
    preds = model.predict(X) # 1 normal, -1 anomaly
    scores = model.decision_function(X) # Higher score = more normal
    
    df_result = df.copy()
    df_result['is_anomaly'] = (preds == -1).astype(int)
    df_result['anomaly_score'] = -scores # Inverse so higher = more anomalous
    return df_result

def explain_anomaly_reasons(row):
    """
    Generate readable explanations for detected movement anomalies.
    """
    reasons = []
    if row.get('calculated_speed_kmh', 0) > 80:
        reasons.append("Abnormally high movement speed")
    if row.get('step_distance_km', 0) > 15:
        reasons.append("Unusually large distance jump")
    if row.get('hour', 12) < 5 or row.get('hour', 12) > 23:
        reasons.append("Late night/early morning movement outside normal hours")
    if row.get('is_weekend', 0) == 1 and row.get('usual_area', '') != row.get('area_name', ''):
        reasons.append("Deviation from weekend routine area")
    
    if not reasons:
        reasons.append("Statistical feature outlier in combined spatial-temporal space")
    return "; ".join(reasons)
