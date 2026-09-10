import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_folium import st_folium

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scoring import compute_search_priority_score
from src.mapping import build_investigation_map
from src.anomaly_detection import explain_anomaly_reasons

# Set Streamlit Page Config
st.set_page_config(
    page_title="AI Missing Person Location Prediction System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: bold;
        color: #1E3A8A;
        border-bottom: 2px solid #3B82F6;
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-left: 5px solid #2563EB;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .priority-very-high { color: #DC2626; font-weight: bold; }
    .priority-high { color: #EA580C; font-weight: bold; }
    .priority-medium { color: #D97706; font-weight: bold; }
    .priority-low { color: #2563EB; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_models():
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    
    kmeans_model = joblib.load(os.path.join(models_dir, 'clustering_model.pkl'))
    cluster_info = joblib.load(os.path.join(models_dir, 'cluster_info.pkl'))
    anomaly_model = joblib.load(os.path.join(models_dir, 'anomaly_model.pkl'))
    anomaly_scaler = joblib.load(os.path.join(models_dir, 'anomaly_scaler.pkl'))
    location_model = joblib.load(os.path.join(models_dir, 'location_model.pkl'))
    markov_model = joblib.load(os.path.join(models_dir, 'markov_model.pkl'))
    encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
    feature_cols = joblib.load(os.path.join(models_dir, 'feature_cols.pkl'))
    hubs = joblib.load(os.path.join(models_dir, 'hubs.pkl'))
    
    return kmeans_model, cluster_info, anomaly_model, anomaly_scaler, location_model, markov_model, encoders, feature_cols, hubs

@st.cache_data
def load_case_datasets():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    cases_path = os.path.join(data_dir, 'synthetic', 'missing_person_cases.csv')
    traj_path = os.path.join(data_dir, 'processed', 'gps_trajectories_processed.csv')
    
    cases_df = pd.read_csv(cases_path) if os.path.exists(cases_path) else None
    traj_df = pd.read_csv(traj_path) if os.path.exists(traj_path) else None
    
    return cases_df, traj_df

def main():
    st.title("CASEFILE: AI-Powered Missing Person Location Prediction System")
    st.caption("Academic Simulation & Investigation Support System | Advanced Machine Learning")

    try:
        kmeans_model, cluster_info, anomaly_model, anomaly_scaler, location_model, markov_model, encoders, feature_cols, hubs = load_all_models()
        cases_df, traj_df = load_case_datasets()
    except Exception as e:
        st.error(f"Error loading models or datasets: {e}")
        st.warning("Please ensure you run `python scripts/generate_and_train.py` first.")
        st.stop()

    # Sidebar: Case Selection & Inputs
    st.sidebar.header("📋 Case Investigation Setup")
    mode = st.sidebar.radio("Select Case Input Mode:", ["Select Pre-loaded Synthetic Case", "Enter Custom Case Details"])

    if mode == "Select Pre-loaded Synthetic Case" and cases_df is not None:
        selected_case_id = st.sidebar.selectbox("Select Case ID:", cases_df['Case_ID'].tolist())
        case_row = cases_df[cases_df['Case_ID'] == selected_case_id].iloc[0].to_dict()
    else:
        case_row = {
            'Case_ID': 'MP-2026-CUSTOM',
            'Person_ID': 'PERSON_CUSTOM',
            'Age_Group': '18–25',
            'Gender': 'Female',
            'Last_Latitude': 17.4065,
            'Last_Longitude': 78.4772,
            'Last_Seen_Time': '18:45',
            'Day': 'Friday',
            'Weather': 'Rain',
            'Usual_Area': 'Area A',
            'Average_Distance': 8.5,
            'Average_Speed': 24.0,
            'Previous_Area': 'Area C',
            'Time_Since_Last_Seen': 6.0
        }

    # Sidebar parameters override / customization
    st.sidebar.subheader("Attributes")
    case_id = case_row['Case_ID']
    age_group = st.sidebar.selectbox("Age Group", ['18–25', '26–35', '36–50', '51+'], index=['18–25', '26–35', '36–50', '51+'].index(case_row['Age_Group']))
    gender = st.sidebar.selectbox("Gender", ['Male', 'Female', 'Other'], index=['Male', 'Female', 'Other'].index(case_row['Gender']))
    day = st.sidebar.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(case_row['Day']))
    weather = st.sidebar.selectbox("Weather", ['Clear', 'Rain', 'Fog', 'Storm'], index=['Clear', 'Rain', 'Fog', 'Storm'].index(case_row['Weather']))
    usual_area = st.sidebar.selectbox("Usual Area", list(hubs.keys()), index=list(hubs.keys()).index(case_row['Usual_Area']))
    prev_area = st.sidebar.selectbox("Previous Area", list(hubs.keys()), index=list(hubs.keys()).index(case_row['Previous_Area']))
    
    last_lat = st.sidebar.number_input("Last Latitude", value=float(case_row['Last_Latitude']), format="%.4f")
    last_lon = st.sidebar.number_input("Last Longitude", value=float(case_row['Last_Longitude']), format="%.4f")
    last_seen_time = st.sidebar.text_input("Last Seen Time (HH:MM)", value=str(case_row['Last_Seen_Time']))
    time_since_seen = st.sidebar.number_input("Hours Since Disappearance", value=float(case_row['Time_Since_Last_Seen']))
    avg_speed = st.sidebar.number_input("Average Speed (km/h)", value=float(case_row['Average_Speed']))
    avg_dist = st.sidebar.number_input("Average Distance (km)", value=float(case_row['Average_Distance']))

    # Prepare features for ML Model
    try:
        hour = int(last_seen_time.split(':')[0])
    except:
        hour = 12

    input_encoded = {
        'Last_Latitude': last_lat,
        'Last_Longitude': last_lon,
        'Hour': hour,
        'Average_Distance': avg_dist,
        'Average_Speed': avg_speed,
        'Time_Since_Last_Seen': time_since_seen,
        'Age_Group': encoders['Age_Group'].transform([age_group])[0],
        'Gender': encoders['Gender'].transform([gender])[0],
        'Day': encoders['Day'].transform([day])[0],
        'Weather': encoders['Weather'].transform([weather])[0],
        'Usual_Area': encoders['Usual_Area'].transform([usual_area])[0],
        'Previous_Area': encoders['Previous_Area'].transform([prev_area])[0]
    }

    input_df = pd.DataFrame([input_encoded])[feature_cols]

    # Predict Location Probabilities
    probs = location_model.predict_proba(input_df)[0]
    raw_classes = location_model.classes_

    # Target Label Encoder decoding
    target_le = encoders.get('Target_Area', None)
    if target_le is not None:
        classes = target_le.inverse_transform(raw_classes)
    else:
        classes = raw_classes

    # Anomaly Detection Test
    is_weekend = 1 if day in ['Saturday', 'Sunday'] else 0
    anom_feats = pd.DataFrame([{
        'calculated_speed_kmh': avg_speed,
        'step_distance_km': avg_dist,
        'hour': hour,
        'is_weekend': is_weekend
    }])[['calculated_speed_kmh', 'step_distance_km', 'hour', 'is_weekend']]
    
    anom_scaled = anomaly_scaler.transform(anom_feats)
    is_anomaly = anomaly_model.predict(anom_scaled)[0] == -1
    anomaly_score = float(-anomaly_model.decision_function(anom_scaled)[0])

    # Markov Route Prediction
    route_chain, route_prob = markov_model.get_route_string(prev_area, max_steps=4)
    route_list, _ = markov_model.predict_next_steps(prev_area, max_steps=4)
    route_coords = [hubs[area][:2] for area in route_list if area in hubs]

    # Calculate Search Priority Rankings
    area_rankings = []
    for cls, p in zip(classes, probs):
        centroid_lat, centroid_lon, area_desc = hubs[cls]

        # Component metrics for score computation
        visit_freq = 0.4 if cls == usual_area else (0.25 if cls == prev_area else 0.1)
        route_sim = 0.4 if cls in route_list else 0.1
        dist_rel = max(0.0, 1.0 - (avg_dist / 30.0))
        time_rel = 0.3 if (hour >= 8 and hour <= 20) else 0.15
        anom_evid = min(1.0, max(0.0, (anomaly_score + 0.5) / 1.0)) if is_anomaly else 0.1

        score_info = compute_search_priority_score(
            ml_prob=p,
            visit_freq=visit_freq,
            route_sim=route_sim,
            dist_rel=dist_rel,
            time_rel=time_rel,
            anomaly_evid=anom_evid
        )

        area_rankings.append({
            'area_name': cls,
            'description': area_desc,
            'lat': centroid_lat,
            'lon': centroid_lon,
            'probability': p,
            'prob_percent': f"{p * 100:.1f}%",
            'score': score_info['total_score'],
            'priority': score_info['priority'],
            'score_breakdown': score_info['components']
        })

    area_rankings = sorted(area_rankings, key=lambda x: x['score'], reverse=True)
    for idx, item in enumerate(area_rankings):
        item['rank'] = idx + 1

    # DASHBOARD TABS
    tab_overview, tab_rankings, tab_map, tab_route, tab_anomaly, tab_xai = st.tabs([
        "📌 Case Summary", "🏆 Location Rankings", "🗺️ Interactive Map", "🛣️ Route Prediction", "⚠️ Anomaly Analysis", "💡 Explainable AI"
    ])

    with tab_overview:
        st.markdown(f"### Casefile Summary: `{case_id}`")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Person ID", case_row.get('Person_ID', 'N/A'))
        col2.metric("Age & Gender", f"{age_group} | {gender}")
        col3.metric("Last Seen Time", f"{last_seen_time} ({day})")
        col4.metric("Disappearance Duration", f"{time_since_seen} hrs ago")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Geographical Context")
            st.write(f"**Last Known Location:** {last_lat:.4f}, {last_lon:.4f}")
            st.write(f"**Frequently Visited Area:** {usual_area} ({hubs[usual_area][2]})")
            st.write(f"**Previously Visited Area:** {prev_area} ({hubs[prev_area][2]})")
            st.write(f"**Weather Condition:** {weather}")
        with c2:
            st.subheader("Top Priority Area Recommendation")
            top1 = area_rankings[0]
            st.success(f"**Primary Search Zone:** {top1['area_name']} ({top1['description']})")
            st.metric("Priority Level", top1['priority'], delta=f"Score: {top1['score']}/100")
            st.write(f"**Predicted ML Probability:** {top1['prob_percent']}")

    with tab_rankings:
        st.markdown("<div class='main-header'>Search Priority Area Rankings</div>", unsafe_allow_html=True)
        st.markdown("Rankings calculated using weighted fusion: ML Probability (30%), Visit Freq (20%), Route Sim (15%), Distance (15%), Time (10%), Anomaly (10%).")
        
        rank_df = pd.DataFrame([{
            'Rank': item['rank'],
            'Area Name': item['area_name'],
            'Description': item['description'],
            'ML Probability': item['prob_percent'],
            'Search Priority Score': f"{item['score']}/100",
            'Priority Band': item['priority'],
            'Latitude': f"{item['lat']:.4f}",
            'Longitude': f"{item['lon']:.4f}"
        } for item in area_rankings])

        st.table(rank_df)

    with tab_map:
        st.markdown("<div class='main-header'>Interactive Geospatial Investigation Map</div>", unsafe_allow_html=True)
        
        frequent_list = [
            {'area_name': k, 'centroid_lat': v[0], 'centroid_lon': v[1], 'visit_count': 45} for k, v in hubs.items()
        ]

        m = build_investigation_map(
            last_known_coords=(last_lat, last_lon),
            frequent_areas=frequent_list,
            predicted_rankings=area_rankings,
            probable_route_coords=route_coords,
            raw_trajectories_df=traj_df
        )

        st_folium(m, width=1100, height=600)

    with tab_route:
        st.markdown("<div class='main-header'>Markov Chain Route Prediction</div>", unsafe_allow_html=True)
        st.subheader(f"Predicted Route from '{prev_area}':")
        st.info(f"**Probable Sequence:** `{route_chain}`")
        st.metric("Markov Sequence Transition Probability", f"{route_prob*100:.1f}%")
        
        st.subheader("Transition Matrix (Markov Model)")
        st.dataframe(markov_model.transition_matrix.style.highlight_max(axis=1, color='lightgreen'))

    with tab_anomaly:
        st.markdown("<div class='main-header'>Movement Anomaly Detection</div>", unsafe_allow_html=True)
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            if is_anomaly:
                st.error("⚠️ ANOMALOUS MOVEMENT PATTERN DETECTED")
            else:
                st.success("✅ NORMAL MOVEMENT PATTERN DETECTED")
            st.metric("Isolation Forest Anomaly Score", f"{anomaly_score:.3f}")
        
        with col_a2:
            st.subheader("Contextual Explanation")
            sample_row = {'calculated_speed_kmh': avg_speed, 'step_distance_km': avg_dist, 'hour': hour, 'is_weekend': is_weekend, 'usual_area': usual_area, 'area_name': prev_area}
            reason_text = explain_anomaly_reasons(sample_row)
            st.write(f"**Reasoning:** {reason_text}")

    with tab_xai:
        st.markdown("<div class='main-header'>Explainable AI (XAI) & Feature Importance</div>", unsafe_allow_html=True)
        st.subheader("Random Forest / XGBoost Global Feature Importances")
        
        if hasattr(location_model, 'feature_importances_'):
            importances = location_model.feature_importances_
            fi_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': importances
            }).sort_values(by='Importance', ascending=True)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(fi_df['Feature'], fi_df['Importance'], color='#3B82F6')
            ax.set_xlabel("Relative Importance")
            ax.set_title("Key Feature Drivers in Location Prediction")
            st.pyplot(fig)

if __name__ == '__main__':
    main()
