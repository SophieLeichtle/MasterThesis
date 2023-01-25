import numpy as np
from soph.utils.frontier_utils import extract_frontiers, sample_around_frontier
from soph.planning.rt_rrt_star.rt_rrt_star import Node
from soph import DEFAULT_FOOTPRINT_RADIUS
from igibson.utils.constants import OccupancyGridState


def closest_frontier(occupancy_map, rt_rrt_star):
    frontier_lines = extract_frontiers(occupancy_map.grid)
    frontiers = []

    for frontier in frontier_lines:
        if len(frontier) < 10:
            continue
        frontier_center_in_map = (np.array(frontier[0]) + np.array(frontier[-1])) / 2
        frontier_center = occupancy_map.px_to_m(frontier_center_in_map)
        frontier_node = Node(frontier_center, np.inf)
        adjacent_nodes = rt_rrt_star.node_clusters.getNodesAdjacent(frontier_node)
        closest_node = rt_rrt_star.getClosestNeighbor(frontier_node, adjacent_nodes)
        if closest_node is None:
            continue
        dist = closest_node.cost() + closest_node.distance_direct(frontier_node)
        frontiers.append((frontier, dist, closest_node))

    frontiers.sort(key=lambda x: x[1])

    for frontier, dist, closest_node in frontiers:

        samples = sample_around_frontier(frontier, occupancy_map)
        center = occupancy_map.px_to_m(
            (np.array(frontier[0]) + np.array(frontier[-1])) / 2
        )
        samples.sort(key=lambda x: np.linalg.norm(center - x[:2]))

        for s in samples:
            if not occupancy_map.check_if_free(
                np.array([s[0], s[1]]), DEFAULT_FOOTPRINT_RADIUS
            ):
                continue
            sample_node = Node(np.array([s[0], s[1]]), np.inf)
            if not rt_rrt_star.lineOfSight(sample_node, closest_node):
                # continue
                print("no line of sight")
            return s, frontier
    return None, None


def goal_from_poi(
    poi, occupancy_map, rt_rrt_star, base_radius=DEFAULT_FOOTPRINT_RADIUS, n=10
):
    poi_in_map = occupancy_map.m_to_px(poi[:2])
    if (
        occupancy_map.grid[int(poi_in_map[0]), int(poi_in_map[1])]
        == OccupancyGridState.UNKNOWN
    ):
        goal, frontier = closest_frontier_poi(poi, occupancy_map, rt_rrt_star)
        return goal, frontier
    if occupancy_map.check_if_free(poi[:2]):
        return poi, []

    r_samples = np.random.uniform(-1, 1, n)
    c_samples = np.random.uniform(-1, 1, n)
    for r in r_samples:
        for c in c_samples:
            sample = poi + base_radius * 1.5 * np.array([r, c, 0])
            if occupancy_map.check_if_free(sample[:2], base_radius):
                return sample, []
    return None, None


def closest_frontier_poi(poi, occupancy_map, rt_rrt_star):
    frontier_lines = extract_frontiers(occupancy_map.grid)
    vec = np.array([np.cos(poi[2]), np.sin(poi[2])])
    frontiers = []

    for frontier in frontier_lines:
        if len(frontier) < 10:
            continue

        min_dist = np.inf
        for frontier_point in frontier:
            frontier_point = occupancy_map.px_to_m(frontier_point)

            px = frontier_point[0]
            py = frontier_point[1]
            x0 = poi[0]
            y0 = poi[1]
            u0 = vec[0]
            v0 = vec[1]

            a = ((px + v0 * py / u0) - (x0 + v0 * y0 / u0)) / (u0 + v0 * v0 / u0)
            dist = (x0 + a * u0 - px) / v0
            if np.abs(dist) < min_dist:
                min_dist = np.abs(dist)

        frontier_center_in_map = (np.array(frontier[0]) + np.array(frontier[-1])) / 2
        frontier_center = occupancy_map.px_to_m(frontier_center_in_map)
        frontier_node = Node(frontier_center, np.inf)
        adjacent_nodes = rt_rrt_star.node_clusters.getNodesAdjacent(frontier_node)
        closest_node = rt_rrt_star.getClosestNeighbor(frontier_node, adjacent_nodes)
        if closest_node is None:
            continue

        frontiers.append((frontier, dist, closest_node))

    frontiers.sort(key=lambda x: x[1])

    for frontier, dist, closest_node in frontiers:

        samples = sample_around_frontier(frontier, occupancy_map)
        center = occupancy_map.px_to_m(
            (np.array(frontier[0]) + np.array(frontier[-1])) / 2
        )
        samples.sort(key=lambda x: np.linalg.norm(center - x[:2]))

        for s in samples:
            if not occupancy_map.check_if_free(
                np.array([s[0], s[1]]), DEFAULT_FOOTPRINT_RADIUS
            ):
                continue
            sample_node = Node(np.array([s[0], s[1]]), np.inf)
            if not rt_rrt_star.lineOfSight(sample_node, closest_node):
                continue
            return s, frontier
    return None, None
