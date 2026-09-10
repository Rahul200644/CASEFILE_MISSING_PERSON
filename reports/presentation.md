# Project Presentation Slides
## CASEFILE: AI-Powered Missing Person Investigation and Location Prediction System

---

### Slide 1: Title & Overview
- **Project Title:** CASEFILE: AI-Powered Missing Person Investigation System
- **Domain:** Advanced Machine Learning & Geospatial Analytics
- **Objective:** Predict probable search areas, reconstruct routes, detect anomalies, and rank search priority zones for missing-person cases.

---

### Slide 2: Problem Statement & Motivation
- **The Challenge:** Investigating missing person cases involves analyzing massive volumes of movement logs, weather data, and past routine habits under tight time constraints.
- **Core Question:** *"Based on historical movement and contextual evidence, which geographical areas should search teams investigate first?"*

---

### Slide 3: System Architecture & Workflow
1. **Data Preprocessing & Feature Engineering:** Timestamp decomposition, Haversine distance, velocity calculation.
2. **Geospatial Clustering (K-Means / DBSCAN):** Extract frequent stay hubs and activity centers.
3. **Anomaly Detection (Isolation Forest):** Flag statistical trajectory outliers.
4. **Supervised Classifier (Random Forest / XGBoost):** Predict target geographical area with Top-1, Top-3, Top-5 accuracy evaluation.
5. **Route Reconstruction (Markov Chain):** Predict transition sequences ($A \rightarrow B \rightarrow C$).
6. **Search Priority Scoring Engine (0-100):** Multi-factor weighted fusion scoring.
7. **Streamlit & Folium GIS Dashboard:** Interactive visualization hub.

---

### Slide 4: Model Performance Highlights
- **Top-1 Location Accuracy:** 89.0% (XGBoost)
- **Top-3 Location Accuracy:** 95.0%
- **Top-5 Location Accuracy:** 100.0%
- **Anomaly Detection Precision:** 95% on synthetic trajectory benchmark.

---

### Slide 5: Interactive Map & Dashboard Features
- **Last Known Location:** Red marker with pulse radius.
- **Priority Search Zones:** Color-coded circles (Red = Very High, Orange = High, Yellow = Medium, Blue = Low).
- **Markov Route Chains:** Directional polyline overlays.
- **XAI Feature Attribution:** SHAP and feature importance bar charts.

---

### Slide 6: Ethical Framework & Summary
- Strictly academic simulation using synthetic datasets.
- Probabilistic support tool designed to assist human decision-making.
