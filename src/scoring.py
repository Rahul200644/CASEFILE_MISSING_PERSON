import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    'ml_prob': 0.30,
    'historical_visit_freq': 0.20,
    'route_similarity': 0.15,
    'distance_relevance': 0.15,
    'time_relevance': 0.10,
    'anomaly_evidence': 0.10
}

def get_priority_band(score):
    """
    Map numerical score (0 to 100) to priority band.
    """
    if score >= 81:
        return 'Very High'
    elif score >= 61:
        return 'High'
    elif score >= 31:
        return 'Medium'
    else:
        return 'Low'

def compute_search_priority_score(
    ml_prob,
    visit_freq,
    route_sim,
    dist_rel,
    time_rel,
    anomaly_evid,
    weights=None
):
    """
    Compute weighted Search Priority Score (0-100 scale) and assign priority band.
    All input parameters expected on a 0.0 to 1.0 scale.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    score = (
        ml_prob * weights['ml_prob'] +
        visit_freq * weights['historical_visit_freq'] +
        route_sim * weights['route_similarity'] +
        dist_rel * weights['distance_relevance'] +
        time_rel * weights['time_relevance'] +
        anomaly_evid * weights['anomaly_evidence']
    ) * 100.0

    score = float(np.clip(score, 0.0, 100.0))
    priority = get_priority_band(score)

    breakdown = {
        'total_score': round(score, 2),
        'priority': priority,
        'components': {
            'ML Probability (30%)': round(ml_prob * weights['ml_prob'] * 100, 2),
            'Visit Frequency (20%)': round(visit_freq * weights['historical_visit_freq'] * 100, 2),
            'Route Similarity (15%)': round(route_sim * weights['route_similarity'] * 100, 2),
            'Distance Relevance (15%)': round(dist_rel * weights['distance_relevance'] * 100, 2),
            'Time Relevance (10%)': round(time_rel * weights['time_relevance'] * 100, 2),
            'Anomaly Evidence (10%)': round(anomaly_evid * weights['anomaly_evidence'] * 100, 2)
        }
    }
    return breakdown
