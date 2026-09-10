import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import joblib

def calculate_top_k_accuracy(model, X, y, k=3):
    """
    Calculate Top-K accuracy for multi-class location prediction.
    """
    probs = model.predict_proba(X)
    classes = model.classes_
    
    top_k_correct = 0
    for i, true_label in enumerate(y):
        top_k_indices = np.argsort(probs[i])[::-1][:k]
        top_k_labels = classes[top_k_indices]
        if true_label in top_k_labels:
            top_k_correct += 1
            
    return top_k_correct / len(y)

def evaluate_location_model(model, X_test, y_test):
    """
    Compute comprehensive classification metrics including Top-1, Top-3, Top-5 accuracy.
    """
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    n_classes = len(np.unique(y_test))
    top1 = acc
    top3 = calculate_top_k_accuracy(model, X_test, y_test, k=min(3, n_classes))
    top5 = calculate_top_k_accuracy(model, X_test, y_test, k=min(5, n_classes))
    
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'top1_accuracy': float(top1),
        'top3_accuracy': float(top3),
        'top5_accuracy': float(top5),
        'confusion_matrix': cm
    }
    return metrics

def train_and_compare_models(X_train, y_train, X_test, y_test):
    """
    Train Random Forest, XGBoost, and KNN models and return performance comparison table.
    """
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss'),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    best_model = None
    best_score = -1.0
    best_name = ""
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_location_model(model, X_test, y_test)
        results[name] = {
            'model': model,
            'metrics': metrics
        }
        if metrics['f1_score'] > best_score:
            best_score = metrics['f1_score']
            best_model = model
            best_name = name
            
    return results, best_name, best_model

def get_prediction_probabilities(model, sample_df, feature_cols):
    """
    Predict probability distribution over target location areas for a case sample.
    """
    probs = model.predict_proba(sample_df[feature_cols])[0]
    classes = model.classes_
    
    rankings = []
    for cls, p in zip(classes, probs):
        rankings.append({
            'area_name': cls,
            'probability': float(p),
            'prob_percent': f"{p * 100:.1f}%"
        })
        
    rankings = sorted(rankings, key=lambda x: x['probability'], reverse=True)
    for i, item in enumerate(rankings):
        item['rank'] = i + 1
    return rankings
