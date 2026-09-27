import pytest
import pandas as pd
from app.etl import clean_data, build_cohort_table

def test_clean_data_leap_year_and_nulls():
    """
    Asserts etl.py handles leap years in tenure calculations and null cancel_dates[cite: 1].
    """
    # Synthetic DataFrame isolating the specific test cases
    df = pd.DataFrame({
        'user_id': ['U1', 'U2'],
        'signup_date': ['2024-02-29', '2023-01-01'],
        'cancel_date': ['2025-02-28', pd.NaT],
        'last_active_date': ['2025-02-28', '2025-03-01'], # Establishes 2025-03-01 as "today"
        'monthly_spend': [50, 100],
        'event_count': [10, 20],
        'plan_tier': ['Pro', 'Enterprise']
    })
    
    clean_df = clean_data(df)
    
    # 1. Tenure calculation correctness across a leap-year boundary[cite: 1]
    # Signup Feb 2024, cancel Feb 2025 must equal exactly 365 days[cite: 1]
    leap_year_user = clean_df[clean_df['user_id'] == 'U1'].iloc[0]
    assert leap_year_user['tenure_days'] == 365
    
    # 2. Correct handling of null cancel_date[cite: 1]
    # Still-active user -> tenure computed against "today" (max date in set)[cite: 1]
    active_user = clean_df[clean_df['user_id'] == 'U2'].iloc[0]
    assert active_user['churned'] == 0
    
    expected_active_days = (pd.to_datetime('2025-03-01') - pd.to_datetime('2023-01-01')).days
    assert active_user['tenure_days'] == expected_active_days

def test_build_cohort_table_shape():
    """
    Asserts build_cohort_table produces correctly shaped cohort pivot tables[cite: 1].
    """
    # Small synthetic DataFrame mimicking the output of clean_data[cite: 1]
    df = pd.DataFrame({
        'user_id': ['U1', 'U2', 'U3', 'U4'],
        'signup_date': pd.to_datetime(['2024-01-01', '2024-01-15', '2024-02-01', '2024-02-10']),
        'tenure_months': [2.5, 0.5, 1.2, 0.0] 
        # Jan Cohort: U1 active up to month 2, U2 up to month 0
        # Feb Cohort: U3 active up to month 1, U4 up to month 0
    })
    
    matrix = build_cohort_table(df)
    
    # 3. Asserts expected number of rows (cohorts) and columns (months-since-signup)[cite: 1]
    # Two distinct signup months (Jan 2024, Feb 2024) = 2 rows
    assert matrix.shape[0] == 2
    
    # Maximum tenure across all users is 2.5 months (index 0, 1, 2) = 3 columns
    assert matrix.shape[1] == 3
    
    # Baseline validation: Month 0 retention must always be 1.0 (100%)
    assert matrix.iloc[0, 0] == 1.0
    assert matrix.iloc[1, 0] == 1.0