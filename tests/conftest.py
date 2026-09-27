import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_df():
    """Provides a synthetic DataFrame for testing ETL and modeling logic."""
    return pd.DataFrame({
        'user_id': ['U1', 'U2', 'U3', 'U4', 'U5'],
        'signup_date': ['2023-01-01', '2024-02-29', '2023-06-01', '2024-01-01', '2024-01-15'],
        'cancel_date': ['2023-03-01', '2025-02-28', pd.NaT, pd.NaT, '2024-01-30'],
        'last_active_date': ['2023-03-01', '2025-02-28', '2024-05-01', '2024-05-01', '2024-01-30'],
        'monthly_spend': [10, 20, 100, 15, 50],
        'event_count': [5, 50, 100, 10, 2],
        'plan_tier': ['Basic', 'Basic', 'Enterprise', 'Basic', 'Pro']
    })