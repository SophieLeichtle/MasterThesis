import numpy as np

from soph.planning.rt_rrt_star.rt_rrt_star import Node


def frontier_distance_simple(frontier, occupancy_map, rt_rrt_star):
    frontier_center_in_map = (np.array(frontier[0]) + np.array(frontier[-1])) / 2
    frontier_center = occupancy_map.px_to_m(frontier_center_in_map)
    frontier_node = Node(frontier_center, np.inf)
    adjacent_nodes = rt_rrt_star.node_clusters.getNodesAdjacent(frontier_node)
    closest_node = rt_rrt_star.getClosestNeighbor(frontier_node, adjacent_nodes)
    if closest_node is None:
        return np.inf, None
    dist = rt_rrt_star.costRecursive(closest_node) + closest_node.distance_direct(
        frontier_node
    )
    return dist, closest_node


def frontier_distance_visible(frontier, occupancy_map, rt_rrt_star, max_iters=5):
    iter = 0
    closest_node = None
    best_dist = np.inf

    frontier_center_in_map = (np.array(frontier[0]) + np.array(frontier[-1])) / 2
    frontier_center = occupancy_map.px_to_m(frontier_center_in_map)
    frontier_node = Node(frontier_center, np.inf)

    while iter < max_iters and closest_node is None:
        adjacent_nodes = rt_rrt_star.node_clusters.getNodesAdjacent(
            frontier_node, min_radius=iter - 1, max_radius=iter
        )
        if len(adjacent_nodes) == 0:
            iter += 1
            continue

        for node in adjacent_nodes:
            dist = rt_rrt_star.costRecursive(node) + node.distance_direct(frontier_node)
            if dist >= best_dist:
                continue
            pos_in_map = occupancy_map.m_to_px(node.position()).astype(np.int32)
            line_of_sight = False
            for point in frontier:
                if occupancy_map.line_of_sight(pos_in_map, np.array(point)):
                    line_of_sight = True
                    break
            if line_of_sight:
                best_dist = dist
                closest_node = node
        iter += 1
    return best_dist, closest_node
