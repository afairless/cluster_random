#! /usr/bin/env python3

from cluster_matrix.cluster_matrix import data_point_pairs_matches, \
        cluster_contingency_table


def test_data_point_pairs_matches():

    from numpy import array as np_array

    cluster_assignment = np_array(
        [1, 1, 1, 2, 2, 
         1, 1, 2, 2, 2, 
         1, 1, 2, 1, 2]).reshape(3, -1).T
    pair_idxs, cluster_pair_agreement = data_point_pairs_matches(
        cluster_assignment)

    test_pair_idxs = [
        (0, 1), 
        (0, 2), 
        (0, 3), 
        (0, 4), 
        (1, 2), 
        (1, 3), 
        (1, 4), 
        (2, 3), 
        (2, 4), 
        (3, 4)]
    assert pair_idxs == test_pair_idxs

    test_cluster_pair_agreement = np_array((
        [1, 1, 1], 
        [1, 0, 0], 
        [0, 0, 1], 
        [0, 0, 0], 
        [1, 0, 0], 
        [0, 0, 1], 
        [0, 0, 0], 
        [0, 1, 0], 
        [0, 1, 1], 
        [1, 1, 0]))
    assert (cluster_pair_agreement == test_cluster_pair_agreement).all()
 
    # test with disagreements represented as -1 instead of zero
 
    pair_idxs, cluster_pair_agreement = data_point_pairs_matches(
        cluster_assignment, False)

    test_pair_idxs = [
        (0, 1), 
        (0, 2), 
        (0, 3), 
        (0, 4), 
        (1, 2), 
        (1, 3), 
        (1, 4), 
        (2, 3), 
        (2, 4), 
        (3, 4)]
    assert pair_idxs == test_pair_idxs

    test_cluster_pair_agreement = np_array((
        [ 1,  1,  1], 
        [ 1, -1, -1], 
        [-1, -1,  1], 
        [-1, -1, -1], 
        [ 1, -1, -1], 
        [-1, -1,  1], 
        [-1, -1, -1], 
        [-1,  1, -1], 
        [-1,  1,  1], 
        [ 1,  1, -1]))
    assert (cluster_pair_agreement == test_cluster_pair_agreement).all()
 
 
    ##################################################

    cluster_assignment = np_array(
        [1, 1, 2, 2, 3, 3, 
         1, 2, 1, 3, 3, 3]).reshape(2, -1).T
    pair_idxs, cluster_pair_agreement = data_point_pairs_matches(
        cluster_assignment)

    test_pair_idxs = [
        (0, 1), 
        (0, 2), 
        (0, 3), 
        (0, 4), 
        (0, 5), 
        (1, 2), 
        (1, 3), 
        (1, 4), 
        (1, 5), 
        (2, 3), 
        (2, 4), 
        (2, 5), 
        (3, 4),
        (3, 5),
        (4, 5)]
    assert pair_idxs == test_pair_idxs

    test_cluster_pair_agreement = np_array((
        [1, 0], 
        [0, 1], 
        [0, 0], 
        [0, 0], 
        [0, 0], 
        [0, 0], 
        [0, 0], 
        [0, 0], 
        [0, 0], 
        [1, 0], 
        [0, 0], 
        [0, 0], 
        [0, 1], 
        [0, 1], 
        [1, 1]))
    assert (cluster_pair_agreement == test_cluster_pair_agreement).all()
 
    # test with disagreements represented as -1 instead of zero
 
    pair_idxs, cluster_pair_agreement = data_point_pairs_matches(
        cluster_assignment, False)

    test_pair_idxs = [
        (0, 1), 
        (0, 2), 
        (0, 3), 
        (0, 4), 
        (0, 5), 
        (1, 2), 
        (1, 3), 
        (1, 4), 
        (1, 5), 
        (2, 3), 
        (2, 4), 
        (2, 5), 
        (3, 4),
        (3, 5),
        (4, 5)]
    assert pair_idxs == test_pair_idxs

    test_cluster_pair_agreement = np_array((
        [ 1, -1], 
        [-1,  1], 
        [-1, -1], 
        [-1, -1], 
        [-1, -1], 
        [-1, -1], 
        [-1, -1], 
        [-1, -1], 
        [-1, -1], 
        [ 1, -1], 
        [-1, -1], 
        [-1, -1], 
        [-1,  1], 
        [-1,  1], 
        [ 1,  1]))
    assert (cluster_pair_agreement == test_cluster_pair_agreement).all()
 
 

def test_cluster_contingency_table():

    from numpy import array as np_array
     
    cluster_assignment = np_array(
        [1, 1, 2, 2, 3, 3, 
         1, 2, 1, 3, 3, 3]).reshape(2, -1).T

    cluster_contingency = cluster_contingency_table(cluster_assignment)
    test_cluster_contingency = np_array((
        [1, 1, 0], 
        [1, 0, 1], 
        [0, 0, 2]))

    assert (cluster_contingency.values == test_cluster_contingency).all()
