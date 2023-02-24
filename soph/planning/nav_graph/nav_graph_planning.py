import numpy as np
import random
from memory_profiler import profile

from soph.planning.frontiers.frontier_utils import (
    frontier_distance_direct,
    frontier_information_gain,
)
from soph.planning.frontiers.frontier_extraction import extract_frontiers
from soph.planning.motion_planning import plan_to_frontier, FrontierSelectionMethod
from soph.planning.nav_graph.nav_graph_utils import (
    distance_to_frontier,
    frontier_distance_simple,
)


@profile
def next_frontier(env, occupancy_map, nav_graph, method, fusion_weights=(0.7, 0.3)):
    frontier_lines = extract_frontiers(occupancy_map.grid)
    frontiers = []

    robot_pos = env.robots[0].get_position()[:2]
    robot_theta = env.robots[0].get_rpy()[2]

    if method is FrontierSelectionMethod.RANDOM:
        for frontier in frontier_lines:
            if len(frontier) < 10:
                continue
            frontiers.append((frontier, np.inf, None))
        random.shuffle(frontiers)
    if method is FrontierSelectionMethod.CLOSEST_EUCLID:
        for frontier in frontier_lines:
            if len(frontier) < 10:
                continue
            dist = frontier_distance_direct(env, occupancy_map, frontier)
            frontiers.append((frontier, dist, None))
        frontiers.sort(key=lambda x: x[1])
    if method is FrontierSelectionMethod.CLOSEST_GRAPH_SIMPLE:
        for frontier in frontier_lines:
            if len(frontier) < 10:
                continue
            dist, closest_node = frontier_distance_simple(
                frontier, robot_pos, occupancy_map, nav_graph
            )
            frontiers.append((frontier, dist, closest_node))
        frontiers.sort(key=lambda x: x[1])
    if method is FrontierSelectionMethod.CLOSEST_GRAPH_VISIBLE:
        for frontier in frontier_lines:
            if len(frontier) < 10:
                continue
            dist, closest_node = distance_to_frontier(
                frontier, robot_pos, nav_graph, occupancy_map
            )
            frontiers.append((frontier, dist, closest_node))
        frontiers.sort(key=lambda x: x[1])
    if method is FrontierSelectionMethod.BESTINFO:
        for frontier in frontier_lines:
            if len(frontier) < 10:
                continue
            gain = frontier_information_gain(occupancy_map, frontier, 3.5)
            frontiers.append((frontier, gain, None))
        frontiers.sort(key=lambda x: x[1], reverse=True)

    if method is FrontierSelectionMethod.FUSION:
        temp = []
        max_gain = 0
        min_gain = np.inf
        max_dist = 0
        min_dist = np.inf
        for frontier in frontier_lines:
            if len(frontier) < 10:
                continue
            gain = frontier_information_gain(occupancy_map, frontier, 3.5)
            max_gain = max(gain, max_gain)
            min_gain = min(gain, min_gain)
            dist = frontier_distance_direct(env, occupancy_map, frontier)
            max_dist = max(dist, max_dist)
            min_dist = min(dist, min_dist)
            temp.append((frontier, dist, gain))

        dist_diff = max_dist - min_dist
        if dist_diff == 0:
            dist_diff = 1
        gain_diff = max_gain - min_gain
        if gain_diff == 0:
            gain_diff = 1
        for frontier, dist, gain in temp:
            f_d = (dist - min_dist) / dist_diff
            f_g = (gain - min_gain) / gain_diff
            cost = fusion_weights[0] * f_d - fusion_weights[1] * f_g
            frontiers.append((frontier, cost, None))
        frontiers.sort(key=lambda x: x[1])

    for frontier, dist, closest_node in frontiers:
        if closest_node is None:
            dist, closest_node = frontier_distance_simple(
                frontier, robot_pos, occupancy_map, nav_graph
            )
        if closest_node is nav_graph.root:
            closest_node = None

        waypoints, plan = plan_from_frontier(
            robot_pos, robot_theta, frontier, closest_node, occupancy_map
        )
        if plan is not None:
            return waypoints, plan, frontier
    return None, None, []


def frontier_plan_with_nav(env, occupancy_map, nav_graph):
    """
    Create a plan for the next point to be navigated to using frontier exploration.
    The next frontier is selected based on shortest navigation distance.
    To calculate distance, the navigation graph is used.
    """
    robot_pos = env.robots[0].get_position()[:2]
    robot_theta = env.robots[0].get_rpy()[2]

    frontier_lines = extract_frontiers(occupancy_map.grid)
    frontiers = []

    for frontier in frontier_lines:
        if len(frontier) < 10:
            continue
        dist, node = distance_to_frontier(frontier, robot_pos, nav_graph, occupancy_map)
        frontiers.append((frontier, dist, node))

    frontiers.sort(key=lambda x: x[1])

    for frontier, dist, node in frontiers:

        waypoints, plan = plan_from_frontier(
            robot_pos, robot_theta, frontier, node, occupancy_map
        )
        if plan is not None:
            return waypoints, plan, frontier
    return None, None, []


def plan_from_frontier(robot_pos, robot_theta, frontier, closest_node, occupancy_map):
    if np.linalg.norm(np.array(frontier[0]) - np.array(frontier[-1])) < 10:
        return None, None
    if closest_node is None:
        plan = plan_to_frontier(
            [robot_pos[0], robot_pos[1], robot_theta], occupancy_map, frontier
        )
        if plan is not None:
            return [], plan
    else:
        waypoints = closest_node.get_path()
        if len(waypoints) < 2:
            current_point = robot_pos
        else:
            current_point = waypoints[-2].position
        next_point = waypoints[-1].position
        theta = np.arctan2(
            next_point[1] - current_point[1], next_point[0] - current_point[0]
        )
        plan = plan_to_frontier(
            [next_point[0], next_point[1], theta], occupancy_map, frontier
        )
        if plan is not None:
            return waypoints, plan
    return None, None
