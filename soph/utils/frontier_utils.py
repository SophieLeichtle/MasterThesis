from random import sample
import numpy as np
from scipy.ndimage import binary_erosion
from igibson.utils.constants import OccupancyGridState

from soph.utils.utils import bbox
from soph import DEFAULT_FOOTPRINT_RADIUS


def extract_frontiers(map_2d, map_2d_old=None, refine=True):
    """
    Extract frontiers from the occupancy map.
    A frontier is a transition from explored free space to unexplored space
    Returns List of Frontier Lines (List of List of 2d Vectors)
    The last two elements of each Line are duplicates of the start and end points of the line

    :param map_2d: 2d occupancy map of type np.array
    """

    known_map = map_2d != OccupancyGridState.UNKNOWN
    eroded_map = binary_erosion(known_map)
    outline = known_map ^ eroded_map
    filtered = outline & (map_2d == OccupancyGridState.FREESPACE)
    if map_2d_old is not None:
        filtered = filtered & (map_2d_old == OccupancyGridState.UNKNOWN)

    lines = []
    while filtered.any():
        rmin, rmax, cmin, cmax = bbox(filtered)
        rinit = rmin
        for c in range(cmin, cmax + 1):
            if filtered[rinit, c] == 1:
                cinit = c
                break

        filtered[rinit, cinit] = 0
        field = [rinit, cinit]
        line = [field]

        new_fields = []
        for x in range(field[0] - 1, field[0] + 2):
            for y in range(field[1] - 1, field[1] + 2):
                if x == field[0] and y == field[1]:
                    continue
                if filtered[x, y] == 1:
                    filtered[x, y] = 0
                    new_fields.append([x, y])
        if len(new_fields) == 0:
            continue
        if len(new_fields) > 2:
            max_dist = 0
            indexes = []
            for i in range(0, len(new_fields) - 1):
                for j in range(i + 1, len(new_fields)):
                    dist = np.linalg.norm(
                        np.array(new_fields[i]) - np.array(new_fields[j])
                    )
                    if dist > max_dist:
                        indexes = [i, j]
                        max_dist = dist
            start1 = new_fields.pop(indexes[0])
            start2 = new_fields.pop(indexes[1] - 1)
            line.extend(new_fields)
        elif len(new_fields) == 2:
            start1 = new_fields[0]
            start2 = new_fields[1]
        else:
            start1 = new_fields[0]
            start2 = None

        new_fields = [start1]
        while len(new_fields) != 0:
            line.extend(new_fields)
            next_new_fields = []
            for field in new_fields:
                for x in range(field[0] - 1, field[0] + 2):
                    for y in range(field[1] - 1, field[1] + 2):
                        if x == field[0] and y == field[1]:
                            continue
                        if filtered[x, y] == 1:
                            filtered[x, y] = 0
                            next_new_fields.append([x, y])
            new_fields = sorted(
                next_new_fields,
                key=lambda x: np.linalg.norm(np.array(line[-1]) - np.array(x)),
            )

        if start2 is not None:
            new_fields = [start2]
            while len(new_fields) != 0:
                line[:0] = new_fields
                next_new_fields = []
                for field in new_fields:
                    for x in range(field[0] - 1, field[0] + 2):
                        for y in range(field[1] - 1, field[1] + 2):
                            if x == field[0] and y == field[1]:
                                continue
                            if filtered[x, y] == 1:
                                filtered[x, y] = 0
                                next_new_fields.append([x, y])
                new_fields = sorted(
                    next_new_fields,
                    key=lambda x: np.linalg.norm(np.array(line[0]) - np.array(x)),
                    reverse=True,
                )
        if refine:
            refined = refine_frontier(line)
            lines.extend(refined)
        else:
            lines.append(line)
    return lines


def refine_frontier(frontier_line, threshold=10):
    """
    Refine frontier by splitting at extrema
    """
    if len(frontier_line) < 10:
        return [frontier_line]
    current_extremes = [frontier_line[0], frontier_line[-1]]

    furthest_point_index = None
    furthest_dist = 0
    unitv = np.array(current_extremes[0]) - np.array(current_extremes[1])
    unitv = unitv / np.linalg.norm(unitv)

    for index, point in enumerate(frontier_line):
        d = point_to_line_dist(point, current_extremes[0], unitv)
        if np.abs(d) > furthest_dist:
            furthest_dist = np.abs(d)
            furthest_point_index = index

    if furthest_dist < threshold:
        return [frontier_line]
    new_frontiers = []
    new_frontiers.extend(
        refine_frontier(frontier_line[: furthest_point_index + 1], threshold)
    )
    new_frontiers.extend(
        refine_frontier(frontier_line[furthest_point_index:], threshold)
    )
    return new_frontiers


def sample_around_frontier(
    frontier_line, occupancy_map, robot_footprint_radius=DEFAULT_FOOTPRINT_RADIUS
):
    """
    Sample points along the frontier. The points must be unoccupied.
    A line is fit along the frontier and the samples are sampled a set distance perpendicular from the line

    :param frontier_line: List of Points making up the frontier
    :param map: occupancy map of type OccupancyGrid2D
    :param robot_footprint_radius: footprint radius of the robot base, used for distance of samples to line
    """
    extremes = [np.array(frontier_line[0]), np.array(frontier_line[-1])]
    unitv = extremes[0] - extremes[1]
    unitv = unitv / np.linalg.norm(unitv)
    unitv = np.array([unitv[1], -unitv[0]])

    center = (extremes[0] + extremes[1]) / 2

    samples = sample(frontier_line, 10)
    dist = 1.4 * robot_footprint_radius * occupancy_map.m_to_pix_ratio

    valid_samples = []
    votes_plus = 0
    votes_minus = 0
    for samp in samples:
        s = samp + 2 * unitv
        if occupancy_map.grid[int(s[0]), int(s[1])] == OccupancyGridState.FREESPACE:
            votes_plus += 1
        else:
            votes_minus += 1
    factor = 1 if votes_plus > votes_minus else -1
    for samp in samples:
        s = samp + factor * dist * unitv
        if occupancy_map.grid[int(s[0]), int(s[1])] == OccupancyGridState.FREESPACE:
            pos = occupancy_map.px_to_m(s)
            theta = np.arctan2(-center[0] + s[0], center[1] - s[1])
            state = np.array([pos[0], pos[1], theta])
            valid_samples.append(state)

    for angle in np.linspace(0, 2 * np.pi, 10):
        v = dist * np.array([np.cos(angle), np.sin(angle)])
        s = extremes[0] + v
        if occupancy_map.line_of_sight(s, center):
            pos = occupancy_map.px_to_m(s)
            theta = np.arctan2(-center[0] + s[0], center[1] - s[1])
            state = np.array([pos[0], pos[1], theta])
            valid_samples.append(state)
        s = extremes[1] + v
        if occupancy_map.line_of_sight(s, center):
            pos = occupancy_map.px_to_m(s)
            theta = np.arctan2(-center[0] + s[0], center[1] - s[1])
            state = np.array([pos[0], pos[1], theta])
            valid_samples.append(state)

    return valid_samples


def point_to_line_dist(point, line_origin, unit_v):
    """
    Calculate shortest distance of point to line
    :param point: point
    :param line_origin, unit_v: line defined by origin and unit vector
    """
    p_x = point[0]
    p_y = point[1]
    x_0 = line_origin[0]
    y_0 = line_origin[1]
    u_0 = unit_v[0]
    v_0 = unit_v[1]
    if u_0 == 0:
        return (x_0 - p_x) / v_0
    if v_0 == 0:
        return (y_0 - p_y) / u_0
    ratio = v_0 / u_0
    a = ((p_x + ratio * p_y) - (x_0 + ratio * y_0)) / (u_0 + ratio * v_0)
    dist = (x_0 + a * u_0 - p_x) / v_0
    return dist


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


# deprecated

# def extract_frontiers(map_2d):
#     """
#     Extract frontiers from the occupancy map.
#     A frontier is a transition from explored free space to unexplored space
#     Returns List of Frontier Lines (List of List of 2d Vectors)
#     The last two elements of each Line are duplicates of the start and end points of the line

#     :param map_2d: 2d occupancy map of type np.array
#     """

#     known_map = map_2d != 0.5
#     eroded_map = binary_erosion(known_map)
#     outline = known_map ^ eroded_map
#     filtered = outline & (map_2d == 1)

#     lines = []
#     while filtered.any():
#         rmin, rmax, cmin, cmax = bbox(filtered)
#         rinit = rmin
#         for c in range(cmin, cmax + 1):
#             if filtered[rinit, c] == 1:
#                 cinit = c
#                 break

#         line = []
#         extremes = []
#         new_fields = [[rinit,cinit]]
#         filtered[rinit,cinit] = 0
#         while len(new_fields) != 0:
#             line.extend(new_fields)
#             next_new_fields = []
#             for field in new_fields:
#                 found = False
#                 for x in range(field[0]-1, field[0]+2):
#                     for y in range(field[1]-1, field[1]+2):
#                         if x == field[0] and y == field[1]: continue
#                         if filtered[x,y] == 1:
#                             found = True
#                             filtered[x,y] = 0
#                             next_new_fields.append([x,y])
#                 if (not found) and len(new_fields) < 3:
#                     extremes.append(field)
#             new_fields = next_new_fields
#         if len(extremes) == 1:
#             extremes.append([rinit,cinit])
#         line.extend(extremes)
#         refined = refine_frontier(line)
#         lines.extend(refined)
#     return lines

# def refine_frontier(frontier_line, threshold = 10):
#     if len(frontier_line) < 10: return [frontier_line]
#     current_extremes = [frontier_line[-1], frontier_line[-2]]

#     furthest_point = None
#     furthest_dist = 0
#     unitv = np.array(current_extremes[0]) - np.array(current_extremes[1])
#     unitv = unitv / np.linalg.norm(unitv)
#     for point in frontier_line:
#         d = point_to_line_dist(point, current_extremes[0], unitv)
#         if np.abs(d) > furthest_dist:
#             furthest_dist = np.abs(d)
#             furthest_point = point
#     if furthest_dist < threshold: return [frontier_line]
#     line1 = []
#     line2 = []
#     for point in frontier_line[:-2]:
#         d = point_to_line_dist(point, furthest_point, [unitv[1], -unitv[0]])
#         if d < 0:
#             line1.append(point)
#         elif d > 0:
#             line2.append(point)
#     if point_to_line_dist(current_extremes[0], furthest_point, [unitv[1], -unitv[0]]) < 0:
#         line1.append(current_extremes[0])
#         line2.append(current_extremes[1])
#     else:
#         line1.append(current_extremes[1])
#         line2.append(current_extremes[0])
#     line1.append(furthest_point)
#     line2.append(furthest_point)
#     newfrontiers = []
#     newfrontiers.extend(refine_frontier(line1, threshold=threshold))
#     newfrontiers.extend(refine_frontier(line2, threshold=threshold))
#     return newfrontiers