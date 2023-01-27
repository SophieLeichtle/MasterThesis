import numpy as np

from soph.planning.frontiers.frontier_extraction import extract_frontiers
from soph.planning.motion_planning import plan_to_frontier


def frontier_plan_with_nav(env, map, nav_graph):
    """
    Create a plan for the next point to be navigated to using frontier exploration.
    The next frontier is selected based on shortest navigation distance.
    To calculate distance, the navigation graph is used.
    """
    robot_pos = env.robots[0].get_position()[:2]
    robot_theta = env.robots[0].get_rpy()[2]

    frontier_lines = extract_frontiers(map.grid)
    frontiers = []

    for frontier in frontier_lines:
        if len(frontier) < 10:
            continue
        dist, node = distance_to_frontier(frontier, robot_pos, nav_graph, map)
        frontiers.append((frontier, dist, node))

    frontiers.sort(key=lambda x: x[1])

    for frontier, dist, node in frontiers:

        waypoints, plan = plan_from_frontier(robot_pos, robot_theta, frontier, node)
        if plan is not None:
            return waypoints, plan, frontier
    return None, None, []


def distance_to_frontier(
    frontier_line, robot_pos, nav_graph, occupancy_map, simplified=False
):
    """
    Calculate distance between robot and frontier line
    """
    robot_pos_in_map = occupancy_map.m_to_px(robot_pos)
    frontier_center_in_map = (
        np.array(frontier_line[0]) + np.array(frontier_line[-1])
    ) / 2
    frontier_center = occupancy_map.px_to_m(frontier_center_in_map)
    # Visible from Robot Position?

    dist_robot = np.linalg.norm(robot_pos - frontier_center)
    if occupancy_map.line_of_sight(robot_pos_in_map, frontier_center_in_map):
        return dist_robot, None

    # Visible from some Node?
    closest_visible_node = nav_graph.get_closest_node(frontier_center, occupancy_map)
    if closest_visible_node is not None and closest_visible_node is not nav_graph.root:
        dist_node = closest_visible_node.get_cost() + np.linalg.norm(
            closest_visible_node.position - frontier_center
        )

        if simplified:
            return dist_node, closest_visible_node

        frontier_v = occupancy_map.px_to_m(
            np.array(frontier_line[0])
        ) - occupancy_map.px_to_m(np.array(frontier_line[-1]))
        frontier_v = frontier_v / np.linalg.norm(frontier_v)
        frontier_v = np.array([frontier_v[1], -frontier_v[0]])

        # Check if Projection onto Orthogonal Vector to Frontier have opposite signs
        # Basically, are the Node and the Robot on the same side of the frontier
        if (
            np.dot(frontier_v, robot_pos - frontier_center)
            * np.dot(frontier_v, closest_visible_node.position - frontier_center)
            < 0
        ):
            return dist_node, closest_visible_node

        # Check if Node Closer To Frontier than Robot Position
        if np.linalg.norm(closest_visible_node.position - frontier_center) < dist_robot:
            return dist_node, closest_visible_node
        else:
            return dist_robot, None

    # Not Visible from Anywhere
    closest_node_direct = nav_graph.get_closest_node(frontier_center)
    dist_direct = closest_node_direct.get_cost() + np.linalg.norm(
        closest_node_direct.position - frontier_center
    )
    if dist_direct < dist_robot and closest_node_direct is not nav_graph.root:
        return dist_direct, closest_node_direct
    return dist_robot, None


def plan_from_frontier(robot_pos, robot_theta, frontier, closest_node):
    if closest_node is None:
        plan = plan_to_frontier(
            [robot_pos[0], robot_pos[1], robot_theta], map, frontier
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
        plan = plan_to_frontier([next_point[0], next_point[1], theta], map, frontier)
        if plan is not None:
            return waypoints, plan
    return None, None
