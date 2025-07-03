import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class RFMClustering:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = None

    def calculate_rfm_and_cluster(self, df, group_by_col, amount_col,
                                  datetime_col):
        """
        Calculate RFM metrics and cluster customers
        into high-risk and non-high-risk groups
        """
        # Make a copy to avoid modifying the original dataframe
        df_rfm = df.copy()

        # Convert datetime column to datetime with UTC
        df_rfm[datetime_col] = pd.to_datetime(df_rfm[datetime_col], utc=True)

        # Set snapshot date as one day after the latest transaction
        snapshot_date = df_rfm[datetime_col].max() + pd.Timedelta(days=1)

        # Calculate RFM metrics
        rfm = df_rfm.groupby(group_by_col).agg({
            datetime_col: lambda x: (snapshot_date - x.max()).days,  # Recency
            group_by_col: 'count',  # Frequency
            amount_col: 'sum'  # Monetary
        }).rename(columns={
            datetime_col: 'Recency',
            group_by_col: 'Frequency',
            amount_col: 'Monetary'
        }).reset_index()

        # Handle negative or zero Monetary values
        # (log transform after shifting)
        rfm['Monetary'] = rfm['Monetary'].apply(
            lambda x: max(x, 0))  # Ensure non-negative
        # Log transform to handle skewness
        rfm['Monetary'] = np.log1p(rfm['Monetary'])

        # Scale RFM features
        self.scaler = StandardScaler()
        rfm_scaled = self.scaler.fit_transform(
            rfm[['Recency', 'Frequency', 'Monetary']])

        # Apply K-Means clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters,
                             random_state=self.random_state)
        rfm['Cluster'] = self.kmeans.fit_predict(rfm_scaled)

        # Analyze clusters to identify high-risk
        # (low Frequency, low Monetary, high Recency)
        cluster_summary = rfm.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            group_by_col: 'count'
        }).rename(columns={group_by_col: 'CustomerCount'}).reset_index()

        # Identify high-risk cluster
        if not cluster_summary.empty:
            high_risk_df = cluster_summary[
                (cluster_summary['Recency'] ==
                 cluster_summary['Recency'].max())
                & (cluster_summary['Frequency'] ==
                   cluster_summary['Frequency'].min())
                & (cluster_summary['Monetary'] == 
                   cluster_summary['Monetary'].min())
            ]
            if not high_risk_df.empty:
                high_risk_cluster = high_risk_df['Cluster'].iloc[0]
            else:
                high_risk_cluster = cluster_summary.loc[
                    cluster_summary['Recency'].idxmax(), 'Cluster'
                ]
        else:
            high_risk_cluster = 0  # Default to cluster 0 if no clusters found

        # Assign is_high_risk label
        rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)

        # Merge is_high_risk back to the original dataframe
        df = df.merge(rfm[[group_by_col, 'is_high_risk']],
                      on=group_by_col, how='left')

        return df, rfm, cluster_summary


# Example usage
if __name__ == "__main__":
    pass  # Replace with actual data loading logic
