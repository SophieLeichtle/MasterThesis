from random import sample
import numpy as np
from igibson.utils.constants import OccupancyGridState

from soph import DEFAULT_FOOTPRINT_RADIUS


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


def frontier_distance_direct(env, occupancy_map, frontier):
    frontier_center_in_map = (np.array(frontier[0]) + np.array(frontier[-1])) / 2
    frontier_center = occupancy_map.px_to_m(frontier_center_in_map)

    robot_pos = env.robots[0].get_position()[:2]
    return np.linalg.norm(frontier_center - robot_pos)


def frontier_information_gain(occupancy_map, frontier, lidar_range):
    frontier_center_in_map = (np.array(frontier[0]) + np.array(frontier[-1])) / 2
    frontier_center = occupancy_map.px_to_m(frontier_center_in_map)
    return occupancy_map.new_information_simple(frontier_center, lidar_range)


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
