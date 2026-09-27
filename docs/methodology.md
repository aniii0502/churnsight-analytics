# Statistical Methodology

## Kaplan-Meier Survival Estimation and Right-Censoring
A naive churn rate calculation is fundamentally flawed because it ignores **right-censored data**—customers who are still active and have not yet churned. Excluding them removes vital information about successful retention, while treating them as "retained forever" skews predictions unrealistically.

The Kaplan-Meier estimator solves this by calculating the probability of surviving past a given time interval, factoring in both churned users and active users up to their current tenure. Active users contribute partial information ("time lived so far") to the curve without distorting the final probabilities, resulting in a highly accurate, monotonically non-increasing survival curve.

## Cohort Retention Matrix
To identify temporal trends, users are grouped by their signup month. The cohort matrix tracks the percentage of users from each cohort who remain active in the subsequent months (Month 0, Month 1, etc.). This highlights whether product updates or seasonal changes improve or degrade long-term retention compared to previous cohorts.

## RFM Clustering
Customer segmentation is driven by a K-Means clustering algorithm applied to engineered RFM features:
* **Recency:** Days since the user's last interaction.
* **Frequency:** Total volume of tracked events.
* **Monetary:** Recurring monthly spend.

Features are normalized using standard scaling before clustering to ensure variables with larger magnitudes (like total spend) do not disproportionately dominate the distance calculations.