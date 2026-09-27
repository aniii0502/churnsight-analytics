import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Reads the CSV, applies an adapter for known public datasets (like Telco),
    and strictly validates that all required columns are present.
    """
    df = pd.read_csv(file_path)
    
    # --- TELCO DATASET ADAPTER ---
    # If we detect the raw Kaggle Telco dataset, synthesize our required schema on the fly
    if 'customerID' in df.columns:
        df = df.rename(columns={
            'customerID': 'user_id',
            'MonthlyCharges': 'monthly_spend',
            'Contract': 'plan_tier'
        })
        
        mock_today = pd.to_datetime('2024-01-01')
        tenure_days = df['tenure'] * 30.436875
        
        # Back-calculate signup date
        df['signup_date'] = mock_today - pd.to_timedelta(tenure_days, unit='D')
        
        # Synthesize variance for Recency and Churn dates (from EDA Notebook 4)
        np.random.seed(42)
        churn_mask = df['Churn'] == 'Yes'
        
        df['cancel_date'] = pd.NaT
        df.loc[churn_mask, 'cancel_date'] = mock_today - pd.to_timedelta(np.random.randint(1, 90, size=churn_mask.sum()), unit='D')
        
        df['last_active_date'] = np.where(
            churn_mask, 
            df['cancel_date'], 
            mock_today - pd.to_timedelta(np.random.randint(0, 30, size=len(df)), unit='D')
        )
        
        df['event_count'] = np.maximum(df['tenure'], 1) 
    # -----------------------------

    # The strict contract your SaaS app requires
    required_columns = [
        'user_id', 'signup_date', 'cancel_date', 
        'plan_tier', 'monthly_spend', 'last_active_date', 'event_count'
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
        
    return df

def clean_data(df):
    """
    Parses dates, handles right-censored users, and calculates leap-year safe tenure[cite: 1].
    """
    df = df.copy()
    
    # 1. Date Parsing[cite: 1]
    date_columns = ['signup_date', 'cancel_date', 'last_active_date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        
    # 2. Right-Censoring Logic[cite: 1]
    # Establish "today" as the maximum observation date across the dataset
    observation_date = df[['signup_date', 'cancel_date', 'last_active_date']].max().max()
    
    # Binary flag: 1 if churned, 0 if active/censored[cite: 1]
    df['churned'] = df['cancel_date'].notnull().astype(int)
    
    # Active users use the observation_date as their cutoff for tenure calculation
    df['effective_end_date'] = df['cancel_date'].fillna(observation_date)
    
    # 3. Tenure Calculation (Leap-year safe)[cite: 1]
    df['tenure_days'] = (df['effective_end_date'] - df['signup_date']).dt.days
    df['tenure_months'] = df['tenure_days'] / 30.436875
    
    return df

def build_cohort_table(df):
    """
    Pivots signups by cohort month and months-since-signup to build the retention matrix[cite: 1].
    """
    df = df.copy()
    df['cohort_month'] = df['signup_date'].dt.to_period('M')
    
    records = []
    # Expand each user's lifespan into individual active months
    for _, row in df.iterrows():
        cohort = row['cohort_month']
        user = row['user_id']
        max_active_month = int(np.floor(row['tenure_months']))
        
        for month_index in range(max_active_month + 1):
            records.append({
                'cohort_month': cohort,
                'month_index': month_index,
                'user_id': user
            })
            
    activity_df = pd.DataFrame(records)
    
    # Group by cohort and month to count active users[cite: 1]
    cohort_counts = activity_df.groupby(['cohort_month', 'month_index'])['user_id'].nunique().reset_index()
    
    # Pivot into a matrix[cite: 1]
    cohort_matrix = cohort_counts.pivot(index='cohort_month', columns='month_index', values='user_id')
    
    # Convert raw counts to retention percentages relative to month 0
    cohort_sizes = cohort_matrix.iloc[:, 0]
    retention_matrix = cohort_matrix.divide(cohort_sizes, axis=0)
    
    return retention_matrix

def compute_rfm(df):
    """
    Engineers Recency, Frequency, and Monetary features for customer segmentation[cite: 1].
    """
    rfm_df = df[['user_id', 'last_active_date', 'event_count', 'monthly_spend']].copy()
    observation_date = rfm_df['last_active_date'].max()
    
    rfm_df['Recency'] = (observation_date - rfm_df['last_active_date']).dt.days
    rfm_df['Frequency'] = rfm_df['event_count']
    rfm_df['Monetary'] = rfm_df['monthly_spend']
    
    return rfm_df[['user_id', 'Recency', 'Frequency', 'Monetary']]