import pandas as pd
from lifelines import KaplanMeierFitter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def fit_kaplan_meier(df):
    """
    Fits a Kaplan-Meier survival curve using the lifelines library, 
    accounting for right-censored data (users who have not churned yet)[cite: 1].
    """
    kmf = KaplanMeierFitter()
    
    # Fit the estimator using the leap-year-safe tenure and binary churn flag
    kmf.fit(
        durations=df['tenure_months'], 
        event_observed=df['churned'], 
        label='SaaS Customer Base'
    )
    
    # Returns the fitted estimator, which inherently contains survival_function_ and confidence_interval_ arrays needed for plotting[cite: 1]
    return kmf

def run_rfm_clustering(rfm_df, n_clusters=3):
    """
    Scales RFM features, fits a K-Means clustering model, and returns 
    the updated DataFrame with cluster labels alongside a per-cluster summary[cite: 1].
    """
    # Isolate the features for scaling (ignoring user_id)
    features = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()
    
    # Scale features so Recency, Frequency, and Monetary operate on the same magnitude[cite: 1]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(features)
    
    # Fit K-Means algorithm[cite: 1]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    # Assign cluster labels back to a copy of the original DataFrame[cite: 1]
    rfm_labeled = rfm_df.copy()
    rfm_labeled['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Calculate the per-cluster summary (mean recency, frequency, monetary)[cite: 1]
    cluster_summary = rfm_labeled.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']
    }).round(2)
    
    # Flatten MultiIndex columns and calculate percentage of total base
    cluster_summary.columns = ['Avg_Recency_Days', 'Avg_Frequency', 'Avg_Monetary_Spend', 'Customer_Count']
    cluster_summary['Pct_of_Base'] = (cluster_summary['Customer_Count'] / len(rfm_labeled) * 100).round(1).astype(str) + '%'
    
    return rfm_labeled, cluster_summary