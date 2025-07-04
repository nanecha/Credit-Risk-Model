import pytest
import pandas as pd
import numpy as np
from src.rfm_clustering import RFMClustering


def test_rfm_calculation():
    """
    Test RFM metric calculation in RFMClustering.==
    """
    # Sample data
    data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [1000.0, -200.0, 500.0],
        'TransactionStartTime': ['2018-11-15T02:18:49Z',
                                 '2018-11-15T02:19:08Z',
                                 '2018-11-15T02:44:21Z']
    })

    # Initialize RFMClustering
    rfm_cluster = RFMClustering(n_clusters=2, random_state=42)

    # Calculate RFM and cluster
    df_with_rfm, rfm, cluster_summary = rfm_cluster.calculate_rfm_and_cluster(
        data, group_by_col='CustomerId', amount_col='Amount',
        datetime_col='TransactionStartTime'
    )

    # Expected RFM values
    expected_rfm = pd.DataFrame({
        'CustomerId': ['C1', 'C2'],
        'Recency': [0, 0],  # Same day transactions
        'Frequency': [2, 1],
        # log1p(1000 - 200), log1p(500)
        'Monetary': [np.log1p(800), np.log1p(500)],
        'Cluster': [0, 1],  # Clusters may vary
        'is_high_risk': [0, 1]  # C2 likely high-risk (lower frequency)
    }, columns=['CustomerId', 'Recency', 'Frequency',
                'Monetary', 'Cluster', 'is_high_risk'])

    # Assertions
    assert rfm.shape[0] == 2, "RFM DataFrame should have 2 rows"
    assert all(col in rfm.columns for col in [
               'CustomerId', 'Recency', 'Frequency', 'Monetary',
               'Cluster', 'is_high_risk'])
    assert np.allclose(rfm['Frequency'], expected_rfm['Frequency'], atol=1e-5)
    assert np.allclose(rfm['Monetary'], expected_rfm['Monetary'], atol=1e-5)


def test_high_risk_assignment():
    """
    Test high-risk cluster assignment in RFMClustering.
    """
    # Sample data
    data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2', 'C3'],
        'Amount': [1000.0, 200.0, 50.0, 5000.0],
        'TransactionStartTime': ['2018-11-15T02:18:49Z',
                                 '2018-11-16T02:19:08Z',
                                 '2018-11-14T02:44:21Z',
                                 '2018-11-17T03:32:55Z']
    })

    # Initialize RFMClustering
    rfm_cluster = RFMClustering(n_clusters=2, random_state=42)

    # Calculate RFM and cluster
    df_with_rfm, rfm, cluster_summary = rfm_cluster.calculate_rfm_and_cluster(
        data, group_by_col='CustomerId', amount_col='Amount',
        datetime_col='TransactionStartTime'
    )

    # Expected: C2 has highest recency (3 days),
    # lowest frequency (1), lowest monetary
    assert rfm[rfm['CustomerId'] ==
               'C2']['is_high_risk'].iloc[0] == 1, "C2 should be high-risk"
    assert 'is_high_risk' in df_with_rfm.columns,
    "is_high_risk column missing in output DataFrame"
    assert df_with_rfm['is_high_risk'].isin(
        [0, 1]).all(), "is_high_risk should be binary"


if __name__ == "__main__":
    pytest.main(["-v"])
