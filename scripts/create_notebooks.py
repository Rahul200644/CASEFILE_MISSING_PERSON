import os
import json

def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def make_cell(cell_type, source):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True) if isinstance(source, str) else source
    }

def main():
    nb_dir = os.path.join(os.path.dirname(__file__), '..', 'notebooks')
    os.makedirs(nb_dir, exist_ok=True)

    # 1. 01_data_collection.ipynb
    nb1 = create_notebook([
        make_cell("markdown", "# Module 1: Data Collection & Synthetic Generation\nThis notebook demonstrates synthetic GPS trajectory dataset generation and missing person case records creation."),
        make_cell("code", "import sys\nsys.path.append('../')\nimport pandas as pd\nimport numpy as np\nfrom scripts.generate_and_train import main\n\nprint('Running Data Generation Pipeline...')"),
        make_cell("code", "df_cases = pd.read_csv('../data/synthetic/missing_person_cases.csv')\ndf_cases.head(10)")
    ])
    with open(os.path.join(nb_dir, '01_data_collection.ipynb'), 'w') as f:
        json.dump(nb1, f, indent=2)

    # 2. 02_data_preprocessing.ipynb
    nb2 = create_notebook([
        make_cell("markdown", "# Module 2: Data Preprocessing\nIncludes coordinate validation, missing value handling, timestamp conversion, and temporal feature extraction."),
        make_cell("code", "import sys\nsys.path.append('../')\nimport pandas as pd\nfrom src.preprocessing import preprocess_gps_data\n\nraw_df = pd.read_csv('../data/raw/gps_trajectories_raw.csv')\nclean_df = preprocess_gps_data(raw_df, timestamp_col='datetime')\nclean_df.head()"),
        make_cell("code", "print('Preprocessed Shape:', clean_df.shape)")
    ])
    with open(os.path.join(nb_dir, '02_data_preprocessing.ipynb'), 'w') as f:
        json.dump(nb2, f, indent=2)

    # 3. 03_eda.ipynb
    nb3 = create_notebook([
        make_cell("markdown", "# Module 3 & 4: Exploratory Data Analysis (EDA)\nAnalyzing temporal patterns, speed distributions, and geographical coordinates."),
        make_cell("code", "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\ndf = pd.read_csv('../data/processed/gps_trajectories_processed.csv')\n\nplt.figure(figsize=(10, 4))\nsns.histplot(df['calculated_speed_kmh'], bins=30, kde=True, color='purple')\nplt.title('Speed Distribution (km/h)')\nplt.show()")
    ])
    with open(os.path.join(nb_dir, '03_eda.ipynb'), 'w') as f:
        json.dump(nb3, f, indent=2)

    # 4. 04_clustering.ipynb
    nb4 = create_notebook([
        make_cell("markdown", "# Module 4: Movement Clustering (K-Means & DBSCAN)\nDiscovering frequented geographical locations and cluster centroids."),
        make_cell("code", "import sys\nsys.path.append('../')\nimport pandas as pd\nfrom src.clustering import perform_kmeans_clustering, perform_dbscan_clustering\n\ndf = pd.read_csv('../data/processed/gps_trajectories_processed.csv')\ndf_clustered, kmeans, info = perform_kmeans_clustering(df, n_clusters=5)\nprint(info)")
    ])
    with open(os.path.join(nb_dir, '04_clustering.ipynb'), 'w') as f:
        json.dump(nb4, f, indent=2)

    # 5. 05_anomaly_detection.ipynb
    nb5 = create_notebook([
        make_cell("markdown", "# Module 5: Movement Anomaly Detection\nDetecting unusual movements using Isolation Forest, LOF, and One-Class SVM."),
        make_cell("code", "import sys\nsys.path.append('../')\nimport pandas as pd\nimport joblib\nfrom src.anomaly_detection import detect_anomalies\n\ndf = pd.read_csv('../data/processed/gps_trajectories_processed.csv')\niso_model = joblib.load('../models/anomaly_model.pkl')\nscaler = joblib.load('../models/anomaly_scaler.pkl')\n\nfeatures = ['calculated_speed_kmh', 'step_distance_km', 'hour', 'is_weekend']\nX_scaled = scaler.transform(df[features].fillna(0.0))\ndf['is_anomaly'] = iso_model.predict(X_scaled) == -1\nprint('Total Anomalies:', df['is_anomaly'].sum())")
    ])
    with open(os.path.join(nb_dir, '05_anomaly_detection.ipynb'), 'w') as f:
        json.dump(nb5, f, indent=2)

    # 6. 06_location_prediction.ipynb
    nb6 = create_notebook([
        make_cell("markdown", "# Module 6: Location Prediction & Model Evaluation\nSupervised classification (Random Forest, XGBoost, KNN) for predicting target geographical area."),
        make_cell("code", "import sys\nsys.path.append('../')\nimport pandas as pd\nimport joblib\nfrom src.prediction import evaluate_location_model\n\ncases_df = pd.read_csv('../data/synthetic/missing_person_cases.csv')\nmodel = joblib.load('../models/location_model.pkl')\nprint('Location Model:', type(model).__name__)")
    ])
    with open(os.path.join(nb_dir, '06_location_prediction.ipynb'), 'w') as f:
        json.dump(nb6, f, indent=2)

    # 7. 07_route_prediction.ipynb
    nb7 = create_notebook([
        make_cell("markdown", "# Module 7: Route Prediction (Markov Chain)\nPredicting probable movement route sequences using first-order transition probabilities."),
        make_cell("code", "import sys\nsys.path.append('../')\nimport joblib\n\nmarkov_model = joblib.load('../models/markov_model.pkl')\nroute, prob = markov_model.get_route_string('Area A', max_steps=4)\nprint('Predicted Route:', route)\nprint('Sequence Probability:', f'{prob*100:.2f}%')")
    ])
    with open(os.path.join(nb_dir, '07_route_prediction.ipynb'), 'w') as f:
        json.dump(nb7, f, indent=2)

    print("All 7 Jupyter notebooks generated in notebooks/")

if __name__ == '__main__':
    main()
