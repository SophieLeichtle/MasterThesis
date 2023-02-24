import numpy as np
from scipy.ndimage import binary_erosion
from igibson.utils.constants import OccupancyGridState

from soph.utils.utils import bbox


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
