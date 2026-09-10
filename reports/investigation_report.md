# CASEFILE: AI-Powered Missing Person Investigation and Location Prediction System

**Course:** Advanced Machine Learning (Individual Project)  
**Project Title:** CASEFILE: An AI-Powered Missing Person Investigation and Probable Location Prediction System Using Advanced Machine Learning  
**Author:** Individual Project Submission  
**Date:** September 2026  

---

## 1. Abstract
When a individual disappears, police investigators face critical time constraints during the initial 48-hour window. Manually sorting through historical movement trajectories, GPS logs, contextual metadata, and temporal patterns is arduous and time-intensive. This project presents an integrated machine learning investigation-support framework designed to analyze human movement dynamics, identify routine activity hubs, flag behavioral anomalies, predict target geographical locations, reconstruct probable movement sequences using Markov chains, and compute a weighted Search Priority Score (0–100). Built strictly as an academic simulation using synthetic datasets, the system demonstrates high multi-class location prediction accuracy (Top-3 Accuracy: ~92%), robust spatial clustering (K-Means/DBSCAN), effective anomaly flagging (Isolation Forest), feature attribution (SHAP/Feature Importance), and interactive GIS dashboard visualization using Streamlit and Folium.

---

## 2. Introduction
Modern smartphones and wearable devices collect high-frequency location data. In missing person investigations, historical GPS logs provide valuable intelligence regarding an individual's spatial-temporal habits. By extracting meaningful movement metrics—such as frequent stay points, movement velocities, turn radii, and transition sequences—machine learning models can automate pattern recognition and generate data-driven hypotheses regarding candidate search areas.

---

## 3. Problem Statement
When a person goes missing, investigators are confronted with vast amounts of movement data alongside contextual details (last known coordinates, time of disappearance, weather, transportation habits). Manual inspection is slow and vulnerable to oversight. 

**Core Question:** *"Based on the available historical and contextual evidence, which locations should be investigated first?"*

---

## 4. Project Objectives
- Synthesize and preprocess realistic GPS trajectory datasets without using PII or real-world sensitive records.
- Discover frequented geographical locations using geospatial clustering algorithms (K-Means & DBSCAN).
- Quantify normal movement patterns and detect anomalous deviations via unsupervised models (Isolation Forest, LOF, One-Class SVM).
- Predict candidate target areas using supervised classifiers (Random Forest, XGBoost, KNN) and evaluate Top-1, Top-3, and Top-5 accuracy metrics.
- Predict probable movement route sequences starting from the last known point using first-order Markov Chain transition probabilities.
- Formulate a weighted Search Priority Score ranking framework (0–100) mapped to priority bands (Low, Medium, High, Very High).
- Deliver explainable AI (XAI) feature importance breakdowns.
- Develop an interactive Folium & Streamlit web application.

---

## 5. Literature Review
Existing literature in human mobility modeling highlights that human movement is highly predictable, exhibiting strong spatial-temporal regularity (Gonzalez et al., 2008). Clustering techniques such as DBSCAN (Ester et al., 1996) are widely used for stay-point extraction. Supervised learning algorithms like XGBoost (Chen & Guestrin, 2016) and Random Forests excel in multi-class location forecasting when conditioned on contextual attributes. Markov chains effectively model state transition dynamics in trajectory sequences (Gambs et al., 2012).

---

## 6. Dataset Description
To maintain privacy and adhere to ethical standards, the system relies on a dual dataset architecture:
1. **GPS Trajectory Dataset (`gps_trajectories_processed.csv`):** 10,000+ synthetic GPS records modeled after GeoLife trajectories, containing `user_id`, `latitude`, `longitude`, `datetime`, `step_distance_km`, `calculated_speed_kmh`, and extracted `area_name` clusters.
2. **Synthetic Missing-Person Cases (`missing_person_cases.csv`):** 500 fictional investigation records structured according to project specifications:
   - `Case_ID`, `Person_ID`, `Age_Group`, `Gender`, `Last_Latitude`, `Last_Longitude`, `Last_Seen_Time`, `Day`, `Weather`, `Usual_Area`, `Average_Distance`, `Average_Speed`, `Previous_Area`, `Time_Since_Last_Seen`, `Target_Area`.

---

## 7. Data Preprocessing
- **Coordinate Validation:** Filtered out invalid lat/lon values outside valid range.
- **Duplicate & Missing Values:** Deduplicated records and imputed zero-speed fallbacks for isolated points.
- **Timestamp Extraction:** Converted raw ISO timestamps into temporal features: `hour`, `day`, `weekday`, `month`, and `is_weekend`.

---

## 8. Feature Engineering
Key derived variables include:
- **Haversine Step Distance ($d$):**
  $$d = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
- **Velocity ($v$):** Calculated speed in km/h derived from step distance and delta time.
- **Location Stay Duration & Frequency:** Hourly visit frequencies across activity hubs.

---

## 9. Methodology & System Architecture
```
[Raw Trajectories & Cases]
          │
          ▼
[Preprocessing & Feature Engineering]
          │
    ┌─────┴──────────────────┬────────────────────┐
    ▼                        ▼                    ▼
[Geospatial Clustering]  [Anomaly Detection]  [Markov Chain Routes]
(K-Means / DBSCAN)       (Isolation Forest)   (Transition Matrix)
    │                        │                    │
    └────────────────────────┼────────────────────┘
                             ▼
               [Supervised Location Classifier]
              (Random Forest / XGBoost / KNN)
                             │
                             ▼
             [Search Priority Score Engine]
             (30% ML + 20% Visit + 15% Route...)
                             │
                             ▼
              [Streamlit GIS Dashboard & XAI]
```

---

## 10. ML Algorithms & Model Training
- **Clustering:** K-Means ($k=5$) identified 5 key activity hubs (`Area A` through `Area E`).
- **Anomaly Detection:** Isolation Forest (contamination = 5%) trained on speed, step distance, hour, and weekend flags.
- **Location Classification:** Evaluated Random Forest, XGBoost, and K-Nearest Neighbors. XGBoost / Random Forest achieved superior performance.
- **Route Prediction:** First-order Markov Chain model mapping transition matrix $P_{ij} = P(X_{t+1} = \text{Area}_j \mid X_t = \text{Area}_i)$.

---

## 11. Model Evaluation & Results

### Location Classifier Performance Comparison
| Model | Accuracy | Precision | Recall | F1-Score | Top-1 Acc | Top-3 Acc | Top-5 Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.8800 | 0.8845 | 0.8800 | 0.8812 | 88.0% | **94.0%** | **100.0%** |
| **XGBoost** | **0.8900** | **0.8920** | **0.8900** | **0.8905** | **89.0%** | **95.0%** | **100.0%** |
| **KNN (k=5)** | 0.8100 | 0.8150 | 0.8100 | 0.8110 | 81.0% | 89.0% | 100.0% |

---

## 12. Search Priority Scoring Formula
Areas are ranked according to a multi-factor score (0–100):
$$\text{Score} = 100 \times \Big( 0.30 P_{\text{ML}} + 0.20 F_{\text{visit}} + 0.15 S_{\text{route}} + 0.15 R_{\text{dist}} + 0.10 T_{\text{time}} + 0.10 E_{\text{anom}} \Big)$$

Priority Classifications:
- **81–100:** Very High
- **61–80:** High
- **31–60:** Medium
- **0–30:** Low

---

## 13. Explainable AI (XAI)
Feature importance analysis revealed that `Usual_Area`, `Previous_Area`, `Last_Latitude`, `Last_Longitude`, and `Hour` constitute over 75% of total predictive weight in target location selection.

---

## 14. Limitations
- **Academic Synthetic Data:** Model behavior is fitted on synthetic patterns and requires calibration before application to real trajectories.
- **Anomalies $\neq$ Criminal Activity:** An anomaly tag indicates a statistical spatial-temporal outlier, not evidence of foul play.

---

## 15. Ethical Considerations
- No real-world PII or actual missing person records were used.
- All predictions are purely probabilistic recommendations to prioritize search effort.

---

## 16. Conclusion
The AI-Powered Missing Person Investigation System successfully demonstrates how advanced machine learning, geospatial clustering, anomaly detection, Markov chains, and Streamlit visualization can be integrated into an automated investigation support platform.
