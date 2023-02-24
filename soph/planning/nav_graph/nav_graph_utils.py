import numpy as np


def frontier_distance_simple(frontier_line, robot_pos, occupancy_map, nav_graph):
    frontier_center_in_map = (
        np.array(frontier_line[0]) + np.array(frontier_line[-1])
    ) / 2
    frontier_center = occupancy_map.px_to_m(frontier_center_in_map)

    dist_robot = np.linalg.norm(robot_pos - frontier_center)

    closest_node_direct = nav_graph.get_closest_node(frontier_center)
    dist_direct = closest_node_direct.get_cost() + np.linalg.norm(
        closest_node_direct.position - frontier_center
    )
    if dist_direct < dist_robot or closest_node_direct is not nav_graph.root:
        return dist_direct, closest_node_direct
    return dist_robot, nav_graph.root


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
        return dist_robot, nav_graph.root

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
            return dist_robot, nav_graph.root

    # Not Visible from Anywhere
    closest_node_direct = nav_graph.get_closest_node(frontier_center)
    dist_direct = closest_node_direct.get_cost() + np.linalg.norm(
        closest_node_direct.position - frontier_center
    )
    if dist_direct < dist_robot and closest_node_direct is not nav_graph.root:
        return dist_direct, closest_node_direct
    return dist_robot, nav_graph.root
