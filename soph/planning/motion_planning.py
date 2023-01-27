from soph.utils.utils import bbox
from soph.planning.frontiers.frontier_utils import (
    sample_around_frontier,
)
from soph.planning.frontiers.frontier_extraction import extract_frontiers
import numpy as np
from igibson.external.pybullet_tools.utils import (
    plan_base_motion_2d,
    set_base_values_with_z,
    set_base_values,
)
import logging
from igibson.utils.constants import OccupancyGridState

import time

from soph import DEFAULT_FOOTPRINT_RADIUS
from soph.planning.plan_base_2d_custom import plan_base_motion_custom


def plan_base_motion(
    robot,
    goal,
    map,
    visualize_planning=False,
    visualize_result=False,
    optimize_iter=0,
    algorithm="birrt",
    robot_footprint_radius=DEFAULT_FOOTPRINT_RADIUS,
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
    x, y, theta = goal

    rmin, rmax, cmin, cmax = bbox(map.grid != 0.5)
    corners = (
        tuple(map.px_to_m(np.asarray([rmax, cmin]))),
        tuple(map.px_to_m(np.asarray([rmin, cmax]))),
    )

    occupancy_range = (map.half_size * 2 + 1) / map.m_to_pix_ratio
    grid_resolution = map.half_size * 2 + 1
    robot_footprint_radius_in_map = int(
        robot_footprint_radius / occupancy_range * grid_resolution
    )

    def pos_to_map(pos):
        return np.around(map.m_to_px(pos)).astype(np.int32)

    path = plan_base_motion_2d(
        robot.get_body_ids()[0],
        [x, y, theta],
        corners,
        map_2d=map.grid,
        occupancy_range=occupancy_range,
        grid_resolution=grid_resolution,
        robot_footprint_radius_in_map=robot_footprint_radius_in_map,
        resolutions=np.array([0.05, 0.05, 2 * np.pi]),
        metric2map=pos_to_map,
        upsampling_factor=2,
        optimize_iter=optimize_iter,
        algorithm=algorithm,
        visualize_planning=visualize_planning,
        visualize_result=visualize_result,
    )

    return path


def teleport(env, point):
    set_base_values(env.robots[0].get_body_ids()[0], point)
    env.simulator.step()


def dry_run_base_plan(env, path):
    """
    Dry run base motion plan by setting the base positions without physics simulation

    :param path: base waypoints or None if no plan can be found
    """
    if path is not None:
        for way_point in path:
            set_base_values_with_z(
                env.robots[0].get_body_ids()[0],
                [way_point[0], way_point[1], way_point[2]],
                z=env.initial_pos_z_offset,
            )
            env.simulator.sync()
            # sleep(0.005) # for animation


def frontier_plan_bestinfo(env, map):
    """
    Create a plan for the next point to be navigated to using frontier exploration.
    The next frontier is selected based on maximum information gain.
    """
    robot_pos = env.robots[0].get_position()[:2]
    robot_theta = env.robots[0].get_rpy()[2]
    best_info = 0
    current_plan = None

    frontier_lines = extract_frontiers(map.grid)
    for line in frontier_lines:
        if len(line) > 10:
            samples = sample_around_frontier(line, map)
            samples.sort(
                key=lambda x: np.linalg.norm(
                    [robot_pos[0] - x[0], robot_pos[1] - x[1], robot_theta - x[2]]
                )
            )
            for s in samples:
                if not map.check_if_free(
                    np.array([s[0], s[1]]), DEFAULT_FOOTPRINT_RADIUS
                ):
                    continue
                new_info = map.check_new_information(
                    np.array([s[0], s[1]]), s[2], 2.5, 1.5 * np.pi
                )
                if new_info > best_info:
                    plan = plan_base_motion(env.robots[0], s, map)
                    if plan is None:
                        continue
                    best_info = new_info
                    current_plan = plan
                    break
    return current_plan


def frontier_plan_shortdistance(env, map, verbose=False):
    """
    Create a plan for the next point to be navigated to using frontier exploration.
    The next frontier is selected based on shortest euclidean distance.
    """
    robot_pos = env.robots[0].get_position()[:2]
    robot_pos_in_map = map.m_to_px(robot_pos)
    frontier_lines = extract_frontiers(map.grid)

    def frontier_center(frontier_line):
        return (np.array(frontier_line[-1]) + np.array(frontier_line[0])) / 2

    frontier_lines.sort(
        key=lambda x: np.linalg.norm(robot_pos_in_map - frontier_center(x))
    )

    for line in frontier_lines:
        if len(line) > 10:
            if verbose:
                logging.info(line)
            samples = sample_around_frontier(line, map)
            center = map.px_to_m(frontier_center(line))
            samples.sort(
                key=lambda x: np.linalg.norm([center[0] - x[0], center[1] - x[1]])
            )
            for s in samples:
                if not map.check_if_free(
                    np.array([s[0], s[1]]), DEFAULT_FOOTPRINT_RADIUS
                ):
                    continue
                if verbose:
                    logging.info(s)
                plan = plan_base_motion(env.robots[0], s, map)
                if plan is not None:
                    return plan, line
    return None, []


def plan_to_frontier(start_config, map, frontier):
    """
    Create a plan to navigate to a specific frontier from a starting configuration.
    """
    samples = sample_around_frontier(frontier, map)
    center = map.px_to_m((np.array(frontier[0]) + np.array(frontier[-1])) / 2)
    samples.sort(key=lambda x: np.linalg.norm(center - x[:2]))
    for s in samples:
        if not map.check_if_free(np.array([s[0], s[1]]), DEFAULT_FOOTPRINT_RADIUS):
            continue
        plan = plan_base_motion_custom(start_config, s, map)
        if plan is not None:
            return plan
    return None


def plan_with_poi(env, map, poi, base_radius=DEFAULT_FOOTPRINT_RADIUS, verbose=False):
    """
    Create a plan to navigate to a point of interest.
    If it is within explored space, the plan is created using sampling.
    If not, the closest frontier is found and a plan is made to navigate to it.
    """
    poi_in_map = map.m_to_px(poi[:2])
    if map.grid[int(poi_in_map[0]), int(poi_in_map[1])] == OccupancyGridState.UNKNOWN:
        plan, frontier = frontier_plan_poi(env, map, poi, base_radius, verbose=verbose)
        return plan, frontier
    plan = sample_plan_poi(env, map, poi, base_radius, verbose=verbose)
    return plan, []


def sample_plan_poi(
    env, map, poi, base_radius=DEFAULT_FOOTPRINT_RADIUS, n=10, verbose=False
):
    """
    Create a plan to navigate to a point of interest by sampling points around the poi.
    """

    if verbose:
        logging.info(poi)
    if map.check_if_free(poi[:2], base_radius):
        plan = plan_base_motion(
            env.robots[0], poi, map, robot_footprint_radius=base_radius
        )
        if plan is not None:
            return plan

    r_samples = np.random.uniform(-1, 1, n)
    c_samples = np.random.uniform(-1, 1, n)
    for r in r_samples:
        for c in c_samples:
            sample = poi + base_radius * 1.5 * np.array([r, c, 0])
            if map.check_if_free(sample[:2], base_radius):
                if verbose:
                    logging.info(sample)
                plan = plan_base_motion(env.robots[0], sample, map)
                if plan is None:
                    continue
                return plan
    return None


def frontier_plan_poi(
    env, map, poi, base_radius=DEFAULT_FOOTPRINT_RADIUS, verbose=False
):
    """
    Find the closest frontier to an unexplored point of interest and return a plan to navigate to it.
    """

    frontier_lines = extract_frontiers(map.grid)
    vec = np.array([np.cos(poi[2]), np.sin(poi[2])])
    current_plan = None

    while current_plan is None and len(frontier_lines) > 0:
        best_dist = np.inf
        best_frontier = None
        for line in frontier_lines:
            if len(line) < 10:
                continue
            min_dist = np.inf
            for frontier_point in line:
                frontier_point = map.px_to_m(frontier_point)

                px = frontier_point[0]
                py = frontier_point[1]
                x0 = poi[0]
                y0 = poi[1]
                u0 = vec[0]
                v0 = vec[1]

                a = ((px + v0 * py / u0) - (x0 + v0 * y0 / u0)) / (u0 + v0 * v0 / u0)
                dist = (x0 + a * u0 - px) / v0
                if np.abs(dist) < best_dist:
                    min_dist = np.abs(dist)
            if min_dist < best_dist:
                best_dist = min_dist
                best_frontier = line

        current_plan = None
        if verbose:
            logging.info(best_frontier)

        center = (np.array(best_frontier[-1]) + np.array(best_frontier[0])) / 2

        samples = sample_around_frontier(best_frontier, map)
        samples.sort(key=lambda x: np.linalg.norm([center[0] - x[0], center[1] - x[1]]))
        for s in samples:
            if not map.check_if_free(np.array([s[0], s[1]]), base_radius):
                continue
            s[2] = poi[2]
            if verbose:
                logging.info(s)
            plan = plan_base_motion(
                env.robots[0], s, map, robot_footprint_radius=base_radius
            )
            if plan is None:
                continue
            current_plan = plan
            break
        if current_plan is None:
            frontier_lines.remove(best_frontier)
    return current_plan, best_frontier


def get_poi(detection, max_depth=2):
    """
    Create a point of interest from a detection
    """
    unitv = np.array([np.cos(detection[2]), np.sin(detection[2])])
    position = np.array([detection[0], detection[1]])
    new_pos = position + max_depth * unitv
    return np.array([new_pos[0], new_pos[1], detection[2]])


# deprecated

# def frontier_plan_detection(env, map, detection):
#     frontier_lines = extract_frontiers(map.grid)
#     vec = np.array([np.cos(detection[2]), np.sin(detection[2])])
#     current_plan = None

#     while current_plan is None and len(frontier_lines) > 0:
#         best_dist = np.inf
#         best_frontier = None
#         for line in frontier_lines:
#             if len(line) < 10: continue

#             min_dist = np.inf
#             for frontier_point in line:
#                 frontier_point = map.px_to_m(frontier_point)

#                 px = frontier_point[0]
#                 py = frontier_point[1]
#                 x0 = detection[0]
#                 y0 = detection[1]
#                 u0 = vec[0]
#                 v0 = vec[1]

#                 a = ((px + v0 * py / u0) - (x0 + v0 * y0 / u0))/(u0 + v0 * v0 / u0)
#                 if a < 0: continue
#                 dist = (x0 + a*u0 - px) / v0
#                 if np.abs(dist) < best_dist:
#                     min_dist = np.abs(dist)
#             if min_dist < best_dist:
#                 best_dist = min_dist
#                 best_frontier = line

#         current_plan = None
#         robot_pos = env.robots[0].get_position()[:2]
#         robot_theta = env.robots[0].get_rpy()[2]
#         samples = sample_around_frontier(best_frontier, map)
#         samples.sort(key=lambda x: np.linalg.norm([robot_pos[0] - x[0], robot_pos[1] - x[1], robot_theta - x[2]]))
#         for s in samples:
#             if not map.check_if_free(np.array([s[0],s[1]]), DEFAULT_FOOTPRINT_RADIUS): continue
#             s[2] = detection[2]
#             plan = plan_base_motion(env.robots[0], s, map)
#             if plan is None: continue
#             current_plan = plan
#             break
#         if current_plan is None:
#             frontier_lines.remove(best_frontier)
#     return current_plan
