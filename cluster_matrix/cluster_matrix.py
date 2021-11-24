#! /usr/bin/env python3

from numpy import ndarray as np_ndarray
from pandas import DataFrame as pd_DataFrame
from typing import Tuple
from scipy.special import comb as choose

 
def data_point_pairs_matches(
    data_point_assignments: np_ndarray, 
    disagreement_as_zero: bool=True) -> Tuple[list, np_ndarray]:
    """

    Given a 'cluster assignment' matrix, returns the corresponding 'cluster 
        pair agreement' matrix


    Cluster assignment matrix:
        each column represents the results of a clustering method
        each row represents a data point
        each cell value is the index of the cluster to which that clustering 
            method assigned that data point

        In this example, the first clustering method (in the first column) 
            assigns the first 3 data points to cluster #1 and the last 2 data 
            points to cluster #2:

                        Clustering method #1    Clustering method #2 
        Data point #1   1                       1
        Data point #2   1                       1
        Data point #3   1                       2
        Data point #4   2                       1
        Data point #5   2                       2


    Cluster pair agreement matrix:
        each column represents the results of a clustering method
        each row represents a pair of data points
        each cell value represents whether the both data points in the pair are
            assigned to the same cluster by the clustering method; assignment 
            to the same cluster, or 'cluster agreement', is designated by a 
            '1'; assignment to different clusters, or 'cluster disagreement',
            is designated by a '2'

        In this example, which corresponds to the example cluster assignment
            matrix above, both clustering methods assign data points #1 and #2
            to the same cluster, and they both assign data points #1 and #5 to
            different clusters; clustering method #1 assigns data points #1 and
            #3 to the same cluster, whereas clustering method #1 assigns data 
            points #1 and #3 to different clusters

                                Clustering method #1    Clustering method #2 
        Data points #1 & #2     1                       1
        Data points #1 & #3     1                       0
        Data points #1 & #4     0                       1
        Data points #1 & #5     0                       0
        Data points #2 & #3     1                       0
        Data points #2 & #4     0                       1
        Data points #2 & #5     0                       0
        Data points #3 & #4     0                       0
        Data points #3 & #5     0                       1
        Data points #4 & #5     1                       0


    Parameters:
        'data_point_assignments' - the cluster assignment matrix
        'disagreement_as_zero' - if 'True', cluster disagreements are 
            represented by zeroes; if 'False', they are represented by negative 
            ones

    Returns:
        'pair_idxs' - indices of pairs of data points for each row of 
            'cluster_pair_agreement'
        'cluster_pair_agreement' - the cluster pair agreement matrix
            -
    """

    from numpy import zeros as np_zeros

    # number of pairs of data points
    pairs_n = int(choose(data_point_assignments.shape[0], 2))

    # initialize cluster pair agreement matrix with zeros
    cluster_pair_agreement = np_zeros((
        pairs_n, data_point_assignments.shape[1]), dtype=int)

    for method_idx in range(data_point_assignments.shape[1]):
        pair_idx = 0
        idx_vector = data_point_assignments[:, method_idx]
        pair_idxs = []
        for i in range(len(idx_vector)):
            for j in range(i+1, len(idx_vector)):
                pair_idxs.append((i, j))
                if idx_vector[i] == idx_vector[j]:
                    cluster_pair_agreement[pair_idx, method_idx] = 1
                pair_idx += 1

    if not disagreement_as_zero:
        from numpy import where as np_where
        cluster_pair_agreement = np_where(
            cluster_pair_agreement < 0.5, -1, cluster_pair_agreement)

    return pair_idxs, cluster_pair_agreement
 

def cluster_contingency_table(cluster_assignment: np_ndarray) -> pd_DataFrame:
    """

    Given a 'cluster assignment' matrix, returns the corresponding 'cluster 
        contingency' matrix


    Cluster assignment matrix:
        each column represents the results of a clustering method
        each row represents a data point
        each cell value is the index of the cluster to which that clustering 
            method assigned that data point

        In this example, the first clustering method (in the first column) 
            assigns the first 2 data points to cluster #1, the next 2 data 
            points to cluster #2, and the last 2 data points to cluster #3:

                        Clustering method #1    Clustering method #2 
        Data point #1   1                       1
        Data point #2   1                       2
        Data point #3   2                       1
        Data point #4   2                       3
        Data point #5   3                       3
        Data point #6   3                       3


    Cluster contingency matrix:
        each row represents a cluster assignment by clustering method #1, so 
            that the sum of the row represents the total number of data points 
            assigned to that cluster by clustering method #1
        each column represents a cluster assignment by clustering method #2, so 
            that the sum of the column represents the total number of data 
            points assigned to that cluster by clustering method #2
        each cell value represents the number of data points assigned by each 
            clustering method to the corresponding cluster (according to row or 
            column)

        In this example, which corresponds to the example cluster assignment
            matrix above, both clustering methods assign data point #1 to 
            cluster #1 and data points #5 and #6 to cluster #3; all these data
            points where the two clustering methods agree appear on the 
            diagonal of the matrix; the two clustering methods disagree on the
            other data points, and those entries appear off-diagonal

                                Clustering method #2
                                Cluster #1  Cluster #2  Cluster #3 
        Clustering method #1
        Cluster #1              1           1           0
        Cluster #2              1           0           1
        Cluster #3              0           0           2


    Returns:
        the cluster contingency matrix
            -
    """
 
    from pandas import crosstab as pd_crosstab

    return pd_crosstab( *[
        cluster_assignment[:, i] for i in range(cluster_assignment.shape[1])] )
