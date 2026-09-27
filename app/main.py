import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import business logic modules
from etl import load_data, clean_data, build_cohort_table, compute_rfm
from survival_models import fit_kaplan_meier, run_rfm_clustering

# 1. Page Config (Must be the first Streamlit command)
st.set_page_config(page_title="ChurnSight Analytics", page_icon="📉", layout="wide")

# 2. Sidebar Controls
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Custom CSV", type=["csv"])
data_source = uploaded_file if uploaded_file else "../data/churn_dataset.csv"

rfm_clusters = st.sidebar.slider("RFM Clusters (K-Means)", min_value=2, max_value=6, value=3)

# 3. Cached Data Pipeline
# We wrap the ETL steps in st.cache_data so they don't re-run on every UI interaction (e.g., slider adjustments)
@st.cache_data(show_spinner="Loading and validating data...")
def fetch_and_clean_data(source):
    df_raw = load_data(source)
    return clean_data(df_raw)

@st.cache_data(show_spinner="Generating cohorts...")
def generate_cohorts(df):
    return build_cohort_table(df)

@st.cache_data(show_spinner="Computing RFM...")
def generate_rfm(df):
    return compute_rfm(df)

st.title("ChurnSight Analytics")
st.markdown("Predictive churn analysis, cohort retention, and RFM customer segmentation.")

try:
    # Execute Pipeline
    df = fetch_and_clean_data(data_source)
    
    # 4. Render KPIs
    total_users = len(df)
    churned_users = df['churned'].sum()
    churn_rate = churned_users / total_users
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{total_users:,}")
    col2.metric("Active Customers", f"{total_users - churned_users:,}")
    col3.metric("Global Churn Rate", f"{churn_rate:.1%}")
    
    st.divider()

    # 5. Render Core Sections
    row1_col1, row1_col2 = st.columns(2)
    
    # --- Survival Curve ---
    with row1_col1:
        st.subheader("Kaplan-Meier Survival Curve")
        kmf = fit_kaplan_meier(df)
        
        # Matplotlib integration requires explicit figure creation in Streamlit
        fig_km, ax_km = plt.subplots(figsize=(8, 5))
        kmf.plot_survival_function(ax=ax_km, color='#1f77b4', linewidth=2)
        ax_km.set_ylim(0, 1.05)
        ax_km.set_xlabel("Tenure (Months)")
        ax_km.set_ylabel("Retention Probability")
        ax_km.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig_km)
        
    # --- Cohort Heatmap ---
    with row1_col2:
        st.subheader("Cohort Retention Matrix")
        cohort_matrix = generate_cohorts(df)
        
        fig_cohort, ax_cohort = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            cohort_matrix.iloc[-12:, :12], # Clip to recent 12 cohorts and 12 months for readability
            annot=True, fmt='.0%', cmap='YlGnBu', 
            vmin=0.0, vmax=1.0, ax=ax_cohort, cbar=False
        )
        ax_cohort.set_xlabel("Months Since Signup")
        ax_cohort.set_ylabel("Cohort Month")
        st.pyplot(fig_cohort)
        
    st.divider()
    
    # --- RFM Segmentation ---
    st.subheader("RFM Customer Segmentation")
    
    rfm_df = generate_rfm(df)
    labeled_rfm, rfm_summary = run_rfm_clustering(rfm_df, n_clusters=rfm_clusters)
    
    col_table, col_plot = st.columns([1, 1.5])
    
    with col_table:
        st.markdown("#### Segment Profiles")
        st.dataframe(rfm_summary, use_container_width=True)
        
    with col_plot:
        st.markdown("#### 3D Segment Distribution")
        fig_rfm = plt.figure(figsize=(8, 5))
        ax_rfm = fig_rfm.add_subplot(111, projection='3d')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for cluster_id in range(rfm_clusters):
            cluster_data = labeled_rfm[labeled_rfm['Cluster'] == cluster_id]
            ax_rfm.scatter(
                cluster_data['Recency'], 
                cluster_data['Frequency'], 
                cluster_data['Monetary'], 
                c=colors[cluster_id], 
                label=f'Cluster {cluster_id}',
                alpha=0.6,
                s=20
            )
        ax_rfm.set_xlabel('Recency (Days)')
        ax_rfm.set_ylabel('Frequency')
        ax_rfm.set_zlabel('Monetary')
        ax_rfm.legend()
        st.pyplot(fig_rfm)

except FileNotFoundError:
    st.error(f"⚠️ Dataset not found at `{data_source}`. Please upload a valid CSV using the sidebar.")
except ValueError as ve:
    st.error(f"⚠️ Data Contract Error: {ve}")
except Exception as e:
    st.error(f"⚠️ An unexpected error occurred: {e}")