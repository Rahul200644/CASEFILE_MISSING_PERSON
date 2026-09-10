import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import preprocess_gps_data
from src.feature_engineering import compute_movement_metrics
from src.clustering import perform_kmeans_clustering, perform_dbscan_clustering
from src.anomaly_detection import train_isolation_forest, train_local_outlier_factor, train_one_class_svm
from src.route_prediction import LocationMarkovChain
from src.prediction import train_and_compare_models
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def main():
    print("=== STEP 1: Synthetic GPS Trajectory & Case Dataset Generation ===")
    np.random.seed(42)
    
    # Base City Center Coordinates (Hyderabad / Representative Urban Center)
    BASE_LAT, BASE_LON = 17.3850, 78.4867

    # 5 Major Activity Hubs / Clusters
    HUBS = {
        'Area A': (17.3850, 78.4867, "Residential Zone A"),
        'Area B': (17.4401, 78.3489, "IT Corridor / Office Park B"),
        'Area C': (17.4065, 78.4772, "Commercial Mall & Market C"),
        'Area D': (17.3616, 78.4747, "Central Bus & Transit Hub D"),
        'Area E': (17.4933, 78.3914, "Suburban Outskirts & Parks E")
    }

    user_ids = [f"USER_{i:03d}" for i in range(1, 21)]
    records = []

    start_date = datetime(2026, 8, 1, 6, 0, 0)

    for user in user_ids:
        curr_time = start_date + timedelta(hours=np.random.randint(0, 48))
        curr_hub_name = np.random.choice(list(HUBS.keys()))

        for _ in range(500): # 500 GPS points per user
            lat_center, lon_center, desc = HUBS[curr_hub_name]
            
            # 90% normal movements around hub, 10% movement between hubs
            if np.random.rand() < 0.90:
                lat = lat_center + np.random.normal(0, 0.005)
                lon = lon_center + np.random.normal(0, 0.005)
                speed = max(0.5, np.random.normal(15.0, 10.0))
            else:
                # Anomaly / Transit movement
                next_hub = np.random.choice(list(HUBS.keys()))
                lat_next, lon_next, _ = HUBS[next_hub]
                alpha = np.random.rand()
                lat = lat_center * (1 - alpha) + lat_next * alpha + np.random.normal(0, 0.01)
                lon = lon_center * (1 - alpha) + lon_next * alpha + np.random.normal(0, 0.01)
                speed = max(5.0, np.random.normal(65.0, 20.0))
                curr_hub_name = next_hub

            curr_time += timedelta(minutes=int(np.random.exponential(15) + 1))
            
            records.append({
                'user_id': user,
                'latitude': lat,
                'longitude': lon,
                'datetime': curr_time,
                'raw_speed': speed
            })

    gps_df = pd.DataFrame(records)
    print(f"Generated {len(gps_df)} raw GPS points across {len(user_ids)} users.")

    # Save Raw Data
    raw_path = os.path.join('data', 'raw', 'gps_trajectories_raw.csv')
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    gps_df.to_csv(raw_path, index=False)

    print("=== STEP 2: Preprocessing & Feature Engineering ===")
    gps_df = preprocess_gps_data(gps_df, timestamp_col='datetime')
    gps_df = compute_movement_metrics(gps_df, user_col='user_id', lat_col='latitude', lon_col='longitude', time_col='datetime')

    # Save Processed Trajectory Data
    proc_path = os.path.join('data', 'processed', 'gps_trajectories_processed.csv')
    os.makedirs(os.path.dirname(proc_path), exist_ok=True)
    gps_df.to_csv(proc_path, index=False)

    print("=== STEP 3: Movement Clustering (K-Means) ===")
    gps_df, kmeans_model, cluster_info = perform_kmeans_clustering(gps_df, n_clusters=5)
    print("Cluster Centroids & Visit Ratios:")
    for cid, cdata in cluster_info.items():
        print(f"  {cdata['area_name']}: Lat {cdata['centroid_lat']:.4f}, Lon {cdata['centroid_lon']:.4f}, Visit Ratio: {cdata['visit_ratio']*100:.1f}%")

    print("=== STEP 4: Anomaly Detection Models ===")
    anomaly_features = ['calculated_speed_kmh', 'step_distance_km', 'hour', 'is_weekend']
    X_anom = gps_df[anomaly_features].fillna(0.0)
    
    scaler_anom = StandardScaler()
    X_anom_scaled = scaler_anom.fit_transform(X_anom)

    iso_forest = train_isolation_forest(X_anom_scaled, contamination=0.05)
    lof_model = train_local_outlier_factor(X_anom_scaled, contamination=0.05)
    ocsvm_model = train_one_class_svm(X_anom_scaled, nu=0.05)

    gps_df['is_anomaly_iso'] = (iso_forest.predict(X_anom_scaled) == -1).astype(int)
    num_anomalies = gps_df['is_anomaly_iso'].sum()
    print(f"Detected {num_anomalies} trajectory anomalies out of {len(gps_df)} records ({num_anomalies/len(gps_df)*100:.2f}%).")

    print("=== STEP 5: Route Prediction (Markov Chain) ===")
    user_sequences = gps_df.groupby('user_id')['area_name'].apply(list).tolist()
    markov_chain = LocationMarkovChain()
    markov_chain.fit(user_sequences)
    print("Markov Chain Transition Matrix:")
    print(markov_chain.transition_matrix.round(2))

    print("=== STEP 6: Synthetic Missing Person Case Records Generation ===")
    cases = []
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weathers = ['Clear', 'Rain', 'Fog', 'Storm']
    age_groups = ['18–25', '26–35', '36–50', '51+']
    genders = ['Male', 'Female', 'Other']
    areas = list(HUBS.keys())

    for c_idx in range(1, 501):
        case_id = f"MP-2026-{c_idx:03d}"
        person_id = f"PERSON_{c_idx:03d}"
        age = np.random.choice(age_groups)
        gender = np.random.choice(genders)
        day = np.random.choice(days)
        weather = np.random.choice(weathers)
        usual_area = np.random.choice(areas)
        previous_area = np.random.choice(areas)
        
        # Target Area correlated with usual_area / previous_area
        if np.random.rand() < 0.65:
            target_area = usual_area
        elif np.random.rand() < 0.85:
            target_area = previous_area
        else:
            target_area = np.random.choice(areas)

        last_lat, last_lon, _ = HUBS[previous_area]
        last_lat += np.random.normal(0, 0.008)
        last_lon += np.random.normal(0, 0.008)

        last_seen_hour = np.random.randint(0, 24)
        last_seen_time = f"{last_seen_hour:02d}:{np.random.randint(0, 60):02d}"
        time_since_seen = round(float(np.random.uniform(1.0, 72.0)), 1)
        avg_dist = round(float(np.random.uniform(2.0, 25.0)), 1)
        avg_speed = round(float(np.random.uniform(10.0, 60.0)), 1)

        cases.append({
            'Case_ID': case_id,
            'Person_ID': person_id,
            'Age_Group': age,
            'Gender': gender,
            'Last_Latitude': last_lat,
            'Last_Longitude': last_lon,
            'Last_Seen_Time': last_seen_time,
            'Day': day,
            'Weather': weather,
            'Usual_Area': usual_area,
            'Average_Distance': avg_dist,
            'Average_Speed': avg_speed,
            'Previous_Area': previous_area,
            'Time_Since_Last_Seen': time_since_seen,
            'Target_Area': target_area
        })

    cases_df = pd.DataFrame(cases)
    synth_path = os.path.join('data', 'synthetic', 'missing_person_cases.csv')
    os.makedirs(os.path.dirname(synth_path), exist_ok=True)
    cases_df.to_csv(synth_path, index=False)
    print(f"Saved {len(cases_df)} synthetic missing-person cases to {synth_path}.")

    print("=== STEP 7: Supervised Location Prediction Model Training ===")
    # Encode categorical features
    encoders = {}
    cat_cols = ['Age_Group', 'Gender', 'Day', 'Weather', 'Usual_Area', 'Previous_Area']
    cases_encoded = cases_df.copy()

    for col in cat_cols:
        le = LabelEncoder()
        cases_encoded[col] = le.fit_transform(cases_encoded[col])
        encoders[col] = le

    # Encode Target Area explicitly as integer
    target_le = LabelEncoder()
    cases_encoded['Target_Area_Code'] = target_le.fit_transform(cases_encoded['Target_Area'])
    encoders['Target_Area'] = target_le

    # Parse hour from Last_Seen_Time
    cases_encoded['Hour'] = cases_encoded['Last_Seen_Time'].apply(lambda x: int(x.split(':')[0]))

    feature_cols = [
        'Last_Latitude', 'Last_Longitude', 'Hour', 'Average_Distance', 'Average_Speed',
        'Time_Since_Last_Seen', 'Age_Group', 'Gender', 'Day', 'Weather', 'Usual_Area', 'Previous_Area'
    ]

    X = cases_encoded[feature_cols]
    y = cases_encoded['Target_Area_Code']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models_comparison, best_name, best_model = train_and_compare_models(X_train, y_train, X_test, y_test)
    
    print("\n--- Location Prediction Model Performance ---")
    for mname, mdata in models_comparison.items():
        m = mdata['metrics']
        print(f"Model: {mname:15s} | Accuracy: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f} | Top-3 Acc: {m['top3_accuracy']:.4f} | Top-5 Acc: {m['top5_accuracy']:.4f}")
    print(f"\nBest Model Selected: {best_name}")

    print("=== STEP 8: Save Trained Models & Artifacts ===")
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(kmeans_model, os.path.join(models_dir, 'clustering_model.pkl'))
    joblib.dump(cluster_info, os.path.join(models_dir, 'cluster_info.pkl'))
    joblib.dump(iso_forest, os.path.join(models_dir, 'anomaly_model.pkl'))
    joblib.dump(scaler_anom, os.path.join(models_dir, 'anomaly_scaler.pkl'))
    joblib.dump(best_model, os.path.join(models_dir, 'location_model.pkl'))
    joblib.dump(markov_chain, os.path.join(models_dir, 'markov_model.pkl'))
    joblib.dump(encoders, os.path.join(models_dir, 'label_encoders.pkl'))
    joblib.dump(feature_cols, os.path.join(models_dir, 'feature_cols.pkl'))
    joblib.dump(HUBS, os.path.join(models_dir, 'hubs.pkl'))

    print("All models successfully trained and serialized to models/!")

if __name__ == '__main__':
    main()
