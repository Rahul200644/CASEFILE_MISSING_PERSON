import os
import sys
import zipfile
import io
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import validate_coordinates, handle_missing_values, extract_time_features
from src.feature_engineering import compute_movement_metrics
from src.clustering import perform_kmeans_clustering
from src.anomaly_detection import train_isolation_forest
from src.route_prediction import LocationMarkovChain
from src.prediction import train_and_compare_models
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Base Indian Metropolitan Center (Hyderabad / Telangana, India)
INDIA_BASE_LAT = 17.3850
INDIA_BASE_LON = 78.4867

# Center of original GeoLife trajectories (Beijing)
GEOLIFE_REF_LAT = 39.9838
GEOLIFE_REF_LON = 116.3238

def transform_to_india_coordinates(lat, lon):
    """
    Project GeoLife trajectory points onto Indian geographical coordinates,
    preserving relative distance, spatial distribution, speed, and time metrics.
    """
    dlat = (lat - GEOLIFE_REF_LAT) * 0.5
    dlon = (lon - GEOLIFE_REF_LON) * 0.5
    
    india_lat = INDIA_BASE_LAT + dlat
    india_lon = INDIA_BASE_LON + dlon
    return india_lat, india_lon

def parse_geolife_zip_india(zip_path, target_records=30000):
    print(f"Extracting 30,000 GeoLife records and projecting onto India map...")
    records = []
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        plt_files = [f for f in z.namelist() if f.endswith('.plt')]
        
        user_file_map = {}
        for fname in plt_files:
            parts = fname.split('/')
            if len(parts) >= 4:
                user_id = f"USER_{int(parts[2]):03d}"
                if user_id not in user_file_map:
                    user_file_map[user_id] = []
                user_file_map[user_id].append(fname)

        for uid, u_files in user_file_map.items():
            if len(records) >= target_records:
                break
                
            for fpath in u_files[:10]:
                if len(records) >= target_records:
                    break
                try:
                    lines = z.read(fpath).decode('utf-8', errors='ignore').splitlines()
                    data_lines = lines[6:]
                    
                    for line in data_lines:
                        if len(records) >= target_records:
                            break
                        parts = line.strip().split(',')
                        if len(parts) >= 7:
                            raw_lat = float(parts[0])
                            raw_lon = float(parts[1])
                            date_str = parts[5].strip()
                            time_str = parts[6].strip()
                            
                            ind_lat, ind_lon = transform_to_india_coordinates(raw_lat, raw_lon)
                            
                            records.append({
                                'user_id': uid,
                                'latitude': ind_lat,
                                'longitude': ind_lon,
                                'datetime': f"{date_str} {time_str}"
                            })
                except Exception:
                    continue

    df = pd.DataFrame(records).iloc[:target_records]
    print(f"Successfully Extracted & Projected {len(df)} records onto India map.")
    return df

def main():
    zip_path = r'C:\Users\rahul\Downloads\archive (2).zip'
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found.")
        return

    # STEP 1: Parse & project onto India
    raw_df = parse_geolife_zip_india(zip_path, target_records=30000)

    # Save raw
    os.makedirs(os.path.join('data', 'raw'), exist_ok=True)
    raw_df.to_csv(os.path.join('data', 'raw', 'gps_trajectories_raw.csv'), index=False)

    # STEP 2: Preprocessing & Feature Engineering
    print("Preprocessing 30,000 Indian trajectory points...")
    proc_df = validate_coordinates(raw_df)
    proc_df = handle_missing_values(proc_df)
    proc_df = extract_time_features(proc_df, timestamp_col='datetime')
    proc_df = compute_movement_metrics(proc_df, user_col='user_id', lat_col='latitude', lon_col='longitude', time_col='datetime')
    
    proc_df = proc_df.iloc[:30000].copy()
    print(f"Final Processed Indian Trajectory Records Count: {len(proc_df)} rows.")

    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    proc_df.to_csv(os.path.join('data', 'processed', 'gps_trajectories_processed.csv'), index=False)

    # STEP 3: K-Means Geospatial Clustering in India
    print("Clustering 30,000 Indian GPS coordinates into 5 activity hubs...")
    proc_df, kmeans_model, cluster_info = perform_kmeans_clustering(proc_df, n_clusters=5)
    
    area_descriptions = {
        'Area A': 'Central Residential Zone, India',
        'Area B': 'IT & Tech Park Corridor, India',
        'Area C': 'Commercial Market & Mall Zone, India',
        'Area D': 'Main Transit & Bus Terminal, India',
        'Area E': 'Suburban Outskirts & Parks, India'
    }

    HUBS = {}
    for cid, cdata in cluster_info.items():
        area_name = cdata['area_name']
        desc = area_descriptions.get(area_name, f"Indian Activity Hub {area_name}")
        HUBS[area_name] = (cdata['centroid_lat'], cdata['centroid_lon'], desc)
        print(f"  {area_name} ({desc}): Lat {cdata['centroid_lat']:.4f}, Lon {cdata['centroid_lon']:.4f} (Visits: {cdata['visit_count']})")

    # STEP 4: Anomaly Detection Models
    print("Training Isolation Forest on Indian trajectory features...")
    anomaly_features = ['calculated_speed_kmh', 'step_distance_km', 'hour', 'is_weekend']
    X_anom = proc_df[anomaly_features].fillna(0.0)
    scaler_anom = StandardScaler()
    X_anom_scaled = scaler_anom.fit_transform(X_anom)
    
    iso_forest = train_isolation_forest(X_anom_scaled, contamination=0.05)
    
    # STEP 5: Markov Chain Route Model
    print("Training Markov Chain model on Indian sequence transitions...")
    user_sequences = proc_df.groupby('user_id')['area_name'].apply(list).tolist()
    markov_chain = LocationMarkovChain()
    markov_chain.fit(user_sequences)

    # STEP 6: Synthetic Missing Person Cases linked to Indian Hubs
    print("Generating 500 missing person cases aligned with Indian hubs...")
    np.random.seed(42)
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

        if np.random.rand() < 0.70:
            target_area = usual_area
        elif np.random.rand() < 0.88:
            target_area = previous_area
        else:
            target_area = np.random.choice(areas)

        last_lat, last_lon, _ = HUBS[previous_area]
        last_lat += np.random.normal(0, 0.005)
        last_lon += np.random.normal(0, 0.005)

        last_seen_hour = np.random.randint(0, 24)
        last_seen_time = f"{last_seen_hour:02d}:{np.random.randint(0, 60):02d}"
        time_since_seen = round(float(np.random.uniform(1.0, 72.0)), 1)
        avg_dist = round(float(np.random.uniform(1.0, 20.0)), 1)
        avg_speed = round(float(np.random.uniform(5.0, 50.0)), 1)

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

    # STEP 7: Supervised Location Prediction Training
    print("Training XGBoost / Random Forest location prediction models...")
    encoders = {}
    cat_cols = ['Age_Group', 'Gender', 'Day', 'Weather', 'Usual_Area', 'Previous_Area']
    cases_encoded = cases_df.copy()

    for col in cat_cols:
        le = LabelEncoder()
        cases_encoded[col] = le.fit_transform(cases_encoded[col])
        encoders[col] = le

    target_le = LabelEncoder()
    cases_encoded['Target_Area_Code'] = target_le.fit_transform(cases_encoded['Target_Area'])
    encoders['Target_Area'] = target_le

    cases_encoded['Hour'] = cases_encoded['Last_Seen_Time'].apply(lambda x: int(x.split(':')[0]))

    feature_cols = [
        'Last_Latitude', 'Last_Longitude', 'Hour', 'Average_Distance', 'Average_Speed',
        'Time_Since_Last_Seen', 'Age_Group', 'Gender', 'Day', 'Weather', 'Usual_Area', 'Previous_Area'
    ]

    X = cases_encoded[feature_cols]
    y = cases_encoded['Target_Area_Code']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models_comp, best_name, best_model = train_and_compare_models(X_train, y_train, X_test, y_test)

    print(f"Location Model Training Complete. Best Selected: {best_name}")

    # STEP 8: Save Model Artifacts
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

    print(f"Successfully projected 30,000 records onto INDIA map and updated all models!")

if __name__ == '__main__':
    main()
