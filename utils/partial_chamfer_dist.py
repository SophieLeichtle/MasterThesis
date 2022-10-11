import open3d as o3d
import numpy as np

def partial_chamfer_dist(measured_tree, ground_truth_tree):

    dists_a = measured_tree.compute_point_cloud_distance(ground_truth_tree)
    threshold = np.max(dists_a)
    dists_b = ground_truth_tree.compute_point_cloud_distance(measured_tree)
    dists_b_filtered = dists_b(dists_b < threshold)

    ratio = dists_b_filtered.len() / dists_b.len()

    partial_chamfer_dist = np.average(dists_a**2) + np.average(dists_b_filtered**2)
    chamfer_dist = np.average(dists_a**2) + np.average(dists_b**2)

    return chamfer_dist, partial_chamfer_dist, ratio


