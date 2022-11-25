import numpy as np
from scipy.ndimage import binary_erosion
from random import sample
from soph.utils.utils import bbox
from soph import DEFAULT_FOOTPRINT_RADIUS
from igibson.utils.constants import OccupancyGridState

def extract_frontiers(map_2d):
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

    lines = []
    while filtered.any():
        rmin, rmax, cmin, cmax = bbox(filtered)
        rinit = rmin
        for c in range(cmin, cmax + 1):
            if filtered[rinit, c] == 1:
                cinit = c
                break
        
        filtered[rinit,cinit] = 0
        field = [rinit, cinit]
        line = [field]

        new_fields = []
        for x in range(field[0]-1, field[0]+2):
            for y in range(field[1]-1, field[1]+2):
                if x == field[0] and y == field[1]: continue
                if filtered[x,y] == 1:
                    filtered[x,y] = 0
                    new_fields.append([x,y])
        if len(new_fields) == 0: continue
        if len(new_fields) > 2:
            max_dist = 0
            indexes = []
            for i in range(0, len(new_fields)-1):
                for j in range(i+1, len(new_fields)):
                    dist = np.linalg.norm(np.array(new_fields[i]) - np.array(new_fields[j]))
                    if dist > max_dist:
                        indexes = [i, j]
                        max_dist = dist
            start1 = new_fields.pop(indexes[0])
            start2 = new_fields.pop(indexes[1])
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
                for x in range(field[0]-1, field[0]+2):
                    for y in range(field[1]-1, field[1]+2):
                        if x == field[0] and y == field[1]: continue
                        if filtered[x,y] == 1:
                            filtered[x,y] = 0
                            next_new_fields.append([x,y])
            new_fields = sorted(next_new_fields, key=lambda x: np.linalg.norm(np.array(line[-1]) - np.array(x)))

        if start2 is not None:
            new_fields = [start2]
            while len(new_fields) != 0:
                line[:0] = new_fields
                next_new_fields = []
                for field in new_fields:
                    for x in range(field[0]-1, field[0]+2):
                        for y in range(field[1]-1, field[1]+2):
                            if x == field[0] and y == field[1]: continue
                            if filtered[x,y] == 1:
                                filtered[x,y] = 0
                                next_new_fields.append([x,y])
                new_fields = sorted(next_new_fields, key=lambda x: np.linalg.norm(np.array(line[0]) - np.array(x)), reverse=True)
        
        refined = refine_frontier(line)
        lines.extend(refined)
    return lines

def refine_frontier(frontier_line, threshold = 10):
    if len(frontier_line) < 10: return[frontier_line]
    current_extremes = [frontier_line[0], frontier_line[-1]]

    furthest_point_index = None
    furthest_dist = 0
    unitv = np.array(current_extremes[0]) - np.array(current_extremes[1])
    unitv = unitv / np.linalg.norm(unitv)

    for index in range(0, len(frontier_line)):
        point = frontier_line[index]
        d = point_to_line_dist(point, current_extremes[0], unitv)
        if np.abs(d) > furthest_dist:
            furthest_dist = np.abs(d)
            furthest_point_index = index
    if furthest_dist < threshold: return [frontier_line]
    new_frontiers = []
    new_frontiers.extend(refine_frontier(frontier_line[:furthest_point_index+1], threshold))
    new_frontiers.extend(refine_frontier(frontier_line[furthest_point_index:], threshold))
    return new_frontiers

def sample_around_frontier(frontier_line, map, robot_footprint_radius=DEFAULT_FOOTPRINT_RADIUS):
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
    dist = 1.4 * robot_footprint_radius * map.m_to_pix_ratio

    valid_samples = []
    votes_plus = 0
    votes_minus = 0
    for samp in samples:
        s = samp + 2 * unitv
        if map.grid[int(s[0]), int(s[1])] == OccupancyGridState.FREESPACE:
            votes_plus += 1
        else:
            votes_minus +=1
    factor = 1 if votes_plus > votes_minus else -1
    for samp in samples:
        s = samp + factor * dist * unitv
        if map.grid[int(s[0]),int(s[1])] == OccupancyGridState.FREESPACE:
            pos = map.px_to_m(s)
            theta = np.arctan2(- center[0] + s[0],center[1] - s[1])
            state = np.array([pos[0], pos[1], theta])
            valid_samples.append(state)

    for angle in np.linspace(0, 2*np.pi, 10):
        v = dist * np.array([np.cos(angle), np.sin(angle)])
        s = extremes[0] + v
        if map.line_of_sight(s, center):
            pos = map.px_to_m(s)
            theta = np.arctan2(- center[0] + s[0],center[1] - s[1])
            state = np.array([pos[0], pos[1], theta])
            valid_samples.append(state)
        s = extremes[1] + v
        if map.line_of_sight(s,center):
            pos = map.px_to_m(s)
            theta = np.arctan2(- center[0] + s[0],center[1] - s[1])
            state = np.array([pos[0], pos[1], theta])
            valid_samples.append(state)

    return valid_samples

def point_to_line_dist(p, line_origin, unit_v):
    px = p[0]
    py = p[1]
    x0 = line_origin[0]
    y0 = line_origin[1]
    u0 = unit_v[0]
    v0 = unit_v[1]
    if u0 == 0:
        return (x0 - px) / v0
    if v0 == 0:
        return (y0 - py) / u0
    ratio = v0 / u0
    a = ((px + ratio * py) - (x0 + ratio * y0))/(u0 + ratio * v0)
    dist = (x0 + a*u0 - px) / v0
    return dist


#deprecated

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