# CASEFILE: AI-Powered Missing Person Investigation and Location Prediction System

An end-to-end Advanced Machine Learning system designed to analyze historical GPS movement trajectories, detect anomalous movements, predict probable geographical areas and route sequences, compute weighted search priority scores, and visualize findings on an interactive GIS Streamlit dashboard.

---

## 📁 Project Directory Structure

```
CASEFILE_MISSING_PERSON/
│
├── data/
│   ├── raw/                      # Raw synthetic GPS trajectories
│   ├── processed/                # Preprocessed trajectories with computed features
│   └── synthetic/                # Synthetic missing-person case records
│
├── notebooks/
│   ├── 01_data_collection.ipynb  # Dataset synthesis and loading
│   ├── 02_data_preprocessing.ipynb # Data cleaning and timestamp extraction
│   ├── 03_eda.ipynb               # Exploratory Data Analysis & visual plots
│   ├── 04_clustering.ipynb        # K-Means & DBSCAN geospatial clustering
│   ├── 05_anomaly_detection.ipynb # Isolation Forest, LOF, One-Class SVM
│   ├── 06_location_prediction.ipynb # Supervised ML & Top-K accuracy evaluation
│   └── 07_route_prediction.ipynb # First-order Markov Chain route transition
│
├── models/
│   ├── clustering_model.pkl      # Serialized K-Means cluster model
│   ├── anomaly_model.pkl         # Serialized Isolation Forest model
│   ├── location_model.pkl        # Serialized XGBoost/RandomForest model
│   ├── markov_model.pkl          # Serialized Markov Chain route model
│   └── label_encoders.pkl        # Categorical feature encoders
│
├── src/
│   ├── preprocessing.py          # GPS coordinate validation & temporal features
│   ├── feature_engineering.py    # Haversine distance, velocity & metrics
│   ├── clustering.py             # K-Means and DBSCAN clustering functions
│   ├── anomaly_detection.py      # Outlier detection models and XAI rules
│   ├── prediction.py             # Supervised location prediction & evaluation
│   ├── route_prediction.py       # Markov chain sequence generator
│   ├── scoring.py                # Search Priority Score calculator
│   └── mapping.py                # Folium interactive map builder
│
├── app/
│   └── app.py                    # Interactive Streamlit Web Application
│
├── scripts/
│   ├── generate_and_train.py     # Main end-to-end dataset generation & model training script
│   └── create_notebooks.py       # Script to generate Jupyter Notebook files
│
├── reports/
│   ├── investigation_report.md   # Comprehensive formal technical report
│   └── presentation.md           # Project presentation slides
│
├── requirements.txt              # Required Python dependencies
└── README.md                     # Project documentation & execution guide
```

---

## ⚙️ Installation & Setup

1. **Navigate to the Project Root:**
   ```bash
   cd C:\Users\rahul\.gemini\antigravity\scratch\CASEFILE_MISSING_PERSON
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Execution Guide

### Step 1: Generate Datasets & Train All ML Models
Run the end-to-end data synthesis and model training pipeline:
```bash
python scripts/generate_and_train.py
```

### Step 2: Generate Jupyter Notebooks
Run the notebook generation script:
```bash
python scripts/create_notebooks.py
```

### Step 3: Launch the Interactive Streamlit Web Application
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Minimum ML Techniques Implemented

1. **Movement Clustering:** K-Means & DBSCAN geospatial clustering to discover activity hubs.
2. **Anomaly Detection:** Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM.
3. **Location Prediction:** XGBoost, Random Forest, and KNN with Top-1, Top-3, and Top-5 accuracy evaluation.
4. **Route Prediction:** First-order Markov Chain model calculating route transition probabilities.
5. **Explainable AI (XAI):** Feature importance and contextual anomaly attribution.

---

## ⚠️ Academic Simulation Disclaimer
This project is an academic simulation built using synthetically generated datasets. It is not intended or calibrated for real-world missing-person decision making.
