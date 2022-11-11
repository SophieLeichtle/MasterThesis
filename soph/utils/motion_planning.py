from soph.utils.utils import bbox
import numpy as np
from igibson.external.pybullet_tools.utils import plan_base_motion_2d, set_base_values_with_z
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
from random import sample

def plan_base_motion(
    robot,
    goal,
    map,
    visualize_planning=False,
    visualize_result=False,
    optimize_iter=0,
    algorithm="birrt",
    robot_footprint_radius=0.3
):
    """
    Plan base motion given a base goal
    :param robot: Robot with which to perform planning
    :param goal: Goal Configuration (x, y, theta)
    :param map: Occupancy Grid (of type OccupancyGrid2D)
    :param visualize_planning: boolean. To visualize the planning process
    :param visualize_result: boolean. to visualize the planning results
    :param optimize_iter: iterations of the optimizer to run after path has been found
    :param algorithm: planning algorithm to use 
    :param robot_footprint_radius: robot footprint in meters
    """
    x,y,theta = goal

    rmin, rmax, cmin, cmax = bbox(map.grid != 0.5)
    corners = (tuple(map.px_to_m(np.asarray([rmax, cmin]))),tuple(map.px_to_m(np.asarray([rmin, cmax]))))
    
    occupancy_range = (map.half_size * 2 + 1) / map.m_to_pix_ratio
    grid_resolution =  map.half_size * 2 + 1
    robot_footprint_radius_in_map = int(robot_footprint_radius / occupancy_range * grid_resolution)

    def pos_to_map(pos):
        return map.m_to_px(pos).astype(np.int32)

    path = plan_base_motion_2d(
        robot.get_body_ids()[0],
        [x,y,theta],
        corners,
        map_2d=map.grid,
        occupancy_range= occupancy_range,
        grid_resolution=grid_resolution,
        robot_footprint_radius_in_map=robot_footprint_radius_in_map,
        resolutions=np.array([0.05, 0.05, 2 * np.pi]),
        metric2map=pos_to_map,
        upsampling_factor=2,
        optimize_iter=optimize_iter,
        algorithm=algorithm,
        visualize_planning=visualize_planning,
        visualize_result=visualize_result
    )

    return path

def teleport(env, point):
    set_base_values_with_z(
        env.robots[0].get_body_ids()[0], point, z = env.initial_pos_z_offset
    )
    env.simulator.step()

def dry_run_base_plan(env, path):
    """
    Dry run base motion plan by setting the base positions without physics simulation

    :param path: base waypoints or None if no plan can be found
    """
    if path is not None:
        for way_point in path:
            set_base_values_with_z(
                env.robots[0].get_body_ids()[0], [way_point[0], way_point[1], way_point[2]], z=env.initial_pos_z_offset
            )
            env.simulator.sync()
            # sleep(0.005) # for animation

def extract_frontiers(map_2d):
    """
    Extract frontiers from the occupancy map. 
    A frontier is a transition from explored free space to unexplored space
    Returns List of Frontier Lines (List of List of 2d Vectors)
    The last two elements of each Line are duplicates of the start and end points of the line

    :param map_2d: 2d occupancy map of type np.array
    """

    known_map = map_2d != 0.5
    eroded_map = binary_erosion(known_map)
    outline = known_map ^ eroded_map
    filtered = outline & (map_2d == 1)

    lines = []
    while filtered.any():
        rmin, rmax, cmin, cmax = bbox(filtered)
        rinit = rmin
        for c in range(cmin, cmax + 1):
            if filtered[rinit, c] == 1:
                cinit = c
                break
        
        line = []
        extremes = []
        new_fields = [[rinit,cinit]]
        filtered[rinit,cinit] = 0
        while len(new_fields) != 0:
            line.extend(new_fields)
            next_new_fields = []
            for field in new_fields:
                found = False
                for x in range(field[0]-1, field[0]+2):
                    for y in range(field[1]-1, field[1]+2):
                        if x == field[0] and y == field[1]: continue
                        if filtered[x,y] == 1:
                            found = True
                            filtered[x,y] = 0
                            next_new_fields.append([x,y])
                if (not found) and len(new_fields) < 3:
                    extremes.append(field)
            new_fields = next_new_fields
        if len(extremes) == 1:
            extremes.append([rinit,cinit])
        line.extend(extremes)
        refined = refine_frontier(line)
        lines.extend(refined)
    return lines

def sample_around_frontier(frontier_line, map, robot_footprint_radius=0.32):
    """
    Sample points along the frontier. The points must be unoccupied.
    A line is fit along the frontier and the samples are sampled a set distance perpendicular from the line

    :param frontier_line: List of Points making up the frontier
    :param map: occupancy map of type OccupancyGrid2D
    :param robot_footprint_radius: footprint radius of the robot base, used for distance of samples to line
    """
    extremes = [np.array(frontier_line[-1]), np.array(frontier_line[-2])]
    unitv = extremes[0] - extremes[1]
    unitv = unitv / np.linalg.norm(unitv)
    unitv = np.array([unitv[1], -unitv[0]])
    angle = np.arctan2(unitv[0], unitv[1])

    samples = sample(frontier_line, 10)
    dist = 1.2 *  robot_footprint_radius * map.m_to_pix_ratio

    valid_samples = []
    for samp in samples:
        s = samp + dist * unitv
        if map.grid[int(s[0]),int(s[1])] == 1:
            pos = map.px_to_m(s)
            theta = angle
            state = np.array([pos[0], pos[1], theta])
            valid_samples.append(state)
        s = samp - dist * unitv
        if map.grid[int(s[0]),int(s[1])] == 1:
            pos = map.px_to_m(s)
            theta = angle + np.pi if angle < 0 else angle - np.pi
            state = np.array([pos[0], pos[1], theta])
            valid_samples.append(state)
    return valid_samples

def plan_with_frontiers(env, map):
    """
    Create a plan for the next point to be navigated to using frontier exploration
    """
    robot_pos = env.robots[0].get_position()[:2]
    robot_theta = env.robots[0].get_rpy()[2]
    best_info = 0
    current_plan = None

    frontier_lines = extract_frontiers(map.grid)
    for line in frontier_lines:
        if len(line) > 10:
            samples = sample_around_frontier(line, map)
            samples.sort(key=lambda x: np.linalg.norm([robot_pos[0] - x[0], robot_pos[1] - x[1], robot_theta - x[2]]))
            for s in samples:
                if not map.check_if_free(np.array([s[0],s[1]]), 0.35): continue
                new_info = map.check_new_information(np.array([s[0],s[1]]), s[2], 2.5, 1.5*np.pi)
                if new_info > best_info:
                    plan = plan_base_motion(env.robots[0], s, map)
                    if plan is None: continue
                    best_info = new_info
                    current_plan = plan
                    break
    return current_plan


def plan_detection_frontier(env, map, detection):
    

    frontier_lines = extract_frontiers(map.grid)
    vec = np.array([np.cos(detection[2]), np.sin(detection[2])])
    current_plan = None

    while current_plan is None and len(frontier_lines) > 0:
        best_dist = np.inf
        best_frontier = None
        for line in frontier_lines:
            if len(line) < 10: continue

            min_dist = np.inf
            for frontier_point in line:
                frontier_point = map.px_to_m(frontier_point)

                px = frontier_point[0]
                py = frontier_point[1]
                x0 = detection[0]
                y0 = detection[1]
                u0 = vec[0]
                v0 = vec[1]

                a = ((px + v0 * py / u0) - (x0 + v0 * y0 / u0))/(u0 + v0 * v0 / u0)
                if a < 0: continue
                dist = (x0 + a*u0 - px) / v0
                if np.abs(dist) < best_dist:
                    min_dist = np.abs(dist)
            if min_dist < best_dist:
                best_dist = min_dist
                best_frontier = line
    
        current_plan = None
        robot_pos = env.robots[0].get_position()[:2]
        robot_theta = env.robots[0].get_rpy()[2]  
        samples = sample_around_frontier(best_frontier, map)
        samples.sort(key=lambda x: np.linalg.norm([robot_pos[0] - x[0], robot_pos[1] - x[1], robot_theta - x[2]]))
        for s in samples:
            if not map.check_if_free(np.array([s[0],s[1]]), 0.35): continue
            s[2] = detection[2]
            plan = plan_base_motion(env.robots[0], s, map)
            if plan is None: continue
            current_plan = plan
            break
        if current_plan is None:
            frontier_lines.remove(best_frontier)
    return current_plan

def refine_frontier(frontier_line, threshold = 5):
    if len(frontier_line) < 10: return [frontier_line]
    current_extremes = [frontier_line[-1], frontier_line[-2]]

    furthest_point = None
    furthest_dist = 0
    unitv = np.array(current_extremes[0]) - np.array(current_extremes[1])
    unitv = unitv / np.linalg.norm(unitv)
    for point in frontier_line:
        d = point_to_line_dist(point, current_extremes[0], unitv)
        if np.abs(d) > furthest_dist:
            furthest_dist = np.abs(d)
            furthest_point = point
    if furthest_dist < threshold: return [frontier_line]
    line1 = []
    line2 = []
    for point in frontier_line:
        d = point_to_line_dist(point, furthest_point, [unitv[1], -unitv[0]])
        if d < 0:
            line1.append(point)
        elif d > 0:
            line2.append(point)
    if point_to_line_dist(current_extremes[0], furthest_point, [unitv[1], -unitv[0]]) < 0:
        line1.append(current_extremes[0])
        line2.append(current_extremes[1])
    else:
        line1.append(current_extremes[1])
        line2.append(current_extremes[0])
    line1.append(furthest_point)
    line2.append(furthest_point)
    newfrontiers = []
    newfrontiers.extend(refine_frontier(line1, threshold=threshold))
    newfrontiers.extend(refine_frontier(line2, threshold=threshold))
    return newfrontiers

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