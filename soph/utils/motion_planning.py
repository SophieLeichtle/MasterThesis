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
    robot_footprint_radius=0.32
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
        new_fields = [[rinit,cinit]]
        filtered[rinit,cinit] = 0
        while len(new_fields) != 0:
            line.extend(new_fields)
            next_new_fields = []
            for field in new_fields:
                for x in range(field[0]-1, field[0]+2):
                    for y in range(field[1]-1, field[1]+2):
                        if filtered[x,y] == 1:
                            filtered[x,y] = 0
                            next_new_fields.append([x,y])
            new_fields = next_new_fields
        lines.append(line)
    return lines

def sample_around_frontier(frontier_line, map, robot_footprint_radius=0.32):
    """
    Sample points along the frontier. The points must be unoccupied.
    A line is fit along the frontier and the samples are sampled a set distance perpendicular from the line

    :param frontier_line: List of Points making up the frontier
    :param map: occupancy map of type OccupancyGrid2D
    :param robot_footprint_radius: footprint radius of the robot base, used for distance of samples to line
    """
    
    line_stack = np.vstack(frontier_line)
    domain = [np.min(line_stack[:,1]),np.max(line_stack[:,1])]
    polynomial = np.polynomial.polynomial.Polynomial.fit(line_stack[:, 1], line_stack[:, 0], 1, domain).convert()
    vec = np.array([-1, polynomial.coef[1]])
    vec = vec / np.linalg.norm(vec)
    angle = np.arctan2(vec[0], vec[1])

    samples = sample(frontier_line, 10)
    dist = 1.2 *  robot_footprint_radius * map.m_to_pix_ratio

    valid_samples = []
    for samp in samples:
        s = samp + dist * vec
        if map.grid[int(s[0]),int(s[1])] == 1:
            pos = map.px_to_m(s)
            theta = angle
            state = np.array([pos[0], pos[1], theta])
            valid_samples.append(state)
        s = samp - dist * vec
        if map.grid[int(s[0]),int(s[1])] == 1:
            pos = map.px_to_m(s)
            theta = angle + np.pi if angle < 0 else angle - np.pi
            state = np.array([pos[0], pos[1], theta])
            valid_samples.append(state)
    return valid_samples