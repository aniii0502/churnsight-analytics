import pytest
import pandas as pd
from app.survival_models import fit_kaplan_meier, run_rfm_clustering

def test_run_rfm_clustering():
    """
    Asserts K-Means returns the expected cluster count on mock RFM data[cite: 1].
    """
    # Mock RFM dataset with distinct behavioral groups to ensure stable clustering[cite: 1]
    rfm_df = pd.DataFrame({
        'user_id': ['U1', 'U2', 'U3', 'U4', 'U5', 'U6'],
        'Recency': [2, 5, 45, 50, 120, 130],
        'Frequency': [50, 45, 10, 12, 1, 2],
        'Monetary': [500, 480, 100, 110, 10, 15]
    })
    
    n_clusters = 3
    labeled_df, summary = run_rfm_clustering(rfm_df, n_clusters=n_clusters)
    
    # Extract labels and verify the set length matches the requested cluster count[cite: 1]
    labels = labeled_df['Cluster'].tolist()
    assert len(set(labels)) == n_clusters
    assert len(summary) == n_clusters

def test_fit_kaplan_meier():
    """
    Asserts survival arrays are well-formed and monotonically non-increasing[cite: 1].
    """
    # Mock DataFrame with explicit right-censored records[cite: 1]
    df = pd.DataFrame({
        'tenure_months': [1.0, 2.5, 3.0, 5.0, 12.0],
        'churned': [1, 1, 0, 1, 0] 
    })
    
    kmf = fit_kaplan_meier(df)
    survival_probs = kmf.survival_function_['SaaS Customer Base'].values
    
    # Verify the array is not empty[cite: 1]
    assert len(survival_probs) > 0
    
    # Verify probabilities never increase over time (curve only goes down or stays flat)[cite: 1]
    is_monotonic = all(x >= y for x, y in zip(survival_probs, survival_probs[1:]))
    assert is_monotonic