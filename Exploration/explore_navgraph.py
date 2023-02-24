import os
import logging
from enum import IntEnum
import yaml
import numpy as np
import time
import csv
import argparse

from soph import configs_path
from soph.environments.custom_env import CustomEnv
from soph.occupancy_grid.occupancy_grid import OccupancyGrid2D
from soph.occupancy_grid.occupancy_utils import (
    spin_and_update,
    refine_map,
    initial_position,
)
from soph.planning.nav_graph.nav_graph import NavGraph


from soph.planning.motion_planning import (
    teleport,
    FrontierSelectionMethod,
    sample_plan_poi,
)

from soph.utils.logging_utils import (
    save_map,
    initiate_logging,
    save_nav_map,
    save_map_combo,
)

from soph.planning.nav_graph.nav_graph_planning import next_frontier

from soph import DEFAULT_FOOTPRINT_RADIUS


class RobotState(IntEnum):
    INIT = 0  # unused for now
    PLANNING = 1
    MOVING = 2
    UPDATING = 3
    END = 4


def main(dir_path, config, frontier_method):
    """
    Create an igibson environment.
    The robot tries to perform a simple frontiers based exploration of the environment.
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    config_filename = os.path.join(configs_path, config + ".yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    logging.info("Frontier Selection Method: %s", frontier_method.name)

    # Create Environment
    logging.info("Creating Environment")
    env = CustomEnv(config_file=config_data, mode="gui_interactive")
    env.reset()

    logging.info("Entering State: INIT")
    current_state = RobotState.INIT

    init_pos = initial_position(config)
    env.land(env.robots[0], init_pos, env.robots[0].get_rpy())

    # Create Map
    occupancy_map = OccupancyGrid2D(half_size=350)

    # Initial Map Update
    spin_and_update(env, occupancy_map)

    robot_pos = env.robots[0].get_position()[:2]
    navigation_graph = NavGraph(np.array(robot_pos))

    current_state = RobotState.PLANNING

    waypoints = None
    current_frontier = []
    current_plan = None
    frontier_plan = None

    map_dir = os.path.join(dir_path, "map")
    os.makedirs(map_dir)
    save_map(map_dir, robot_pos, occupancy_map)
    combo_dir = os.path.join(dir_path, "combo")
    os.makedirs(combo_dir)
    save_map_combo(combo_dir, robot_pos, occupancy_map, navigation_graph)
    nav_dir = os.path.join(dir_path, "nav")
    os.makedirs(nav_dir)
    save_nav_map(nav_dir, occupancy_map, navigation_graph)

    exploration_stats_file = os.path.join(dir_path, "exploration_stats.csv")
    planning_stats_file = os.path.join(dir_path, "planning_stats.csv")

    logging.info("Entering State: PLANNING")

    planning_attempts = 0
    max_planning_attempts = 10

    total_distance = 0
    with open(exploration_stats_file, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        fields = ["Distance", "Explored"]
        csvwriter.writerow(fields)
        entry = [total_distance, occupancy_map.explored_space()]
        csvwriter.writerow(entry)
    with open(planning_stats_file, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        fields = ["Type", "Time", "Found"]
        csvwriter.writerow(fields)

    start_time = time.process_time()
    while True:
        # cap at 30 minutes
        if time.process_time() - start_time > 1800:
            current_state = RobotState.END
        if current_state is RobotState.PLANNING:

            planning_attempts += 1
            robot_pos = env.robots[0].get_position()[:2]

            if waypoints is None:
                logging.info("Planning with Frontiers")
                current_plan = None
                t = time.process_time()
                waypoints, frontier_plan, current_frontier = next_frontier(
                    env,
                    occupancy_map,
                    navigation_graph,
                    frontier_method,
                )
                t = time.process_time() - t
                with open(planning_stats_file, "a") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    entry = ["frontier", t, waypoints != None]
                    csvwriter.writerow(entry)

            if waypoints is not None:
                if len(waypoints) == 0:
                    logging.info("Final Waypoint Reached, Moving To Frontier")
                    current_plan = frontier_plan
                    frontier_plan = None
                    waypoints = None
                else:
                    logging.info("Planning to Next Waypoint")
                    next_point = waypoints[0].position
                    theta = np.arctan2(
                        next_point[1] - robot_pos[1], next_point[0] - robot_pos[0]
                    )
                    t = time.process_time()
                    current_plan = sample_plan_poi(
                        env,
                        occupancy_map,
                        np.array([next_point[0], next_point[1], theta]),
                    )
                    t = time.process_time() - t
                    with open(planning_stats_file, "a") as csvfile:
                        csvwriter = csv.writer(csvfile)
                        entry = ["waypoint", t, current_plan != None]
                        csvwriter.writerow(entry)

            if current_plan is not None:
                planning_attempts = 0
                current_state = RobotState.MOVING
                logging.info("Entering State: Moving")
                save_map(
                    map_dir,
                    robot_pos,
                    occupancy_map,
                    current_plan=current_plan,
                    frontier_line=current_frontier,
                )
                save_map_combo(
                    combo_dir,
                    robot_pos,
                    occupancy_map,
                    navigation_graph,
                    current_plan=current_plan,
                    frontier_line=current_frontier,
                )
            else:
                if planning_attempts > max_planning_attempts:
                    logging.warning("Max Planning Attempts reached")
                    logging.info("Entering State: End")
                    current_state = RobotState.END

        elif current_state is RobotState.MOVING:
            current_point = current_plan.pop(0)
            robot_pos = env.robots[0].get_position()[:2]
            total_distance += np.linalg.norm(robot_pos - current_point[:2])

            teleport(env, current_point)

            robot_pos = env.robots[0].get_position()[:2]

            navigation_graph.update_with_robot_pos(robot_pos, occupancy_map, 1.5)
            if len(current_plan) == 0:
                current_plan = None
                if waypoints is None:
                    current_state = RobotState.UPDATING
                    logging.info("Current Plan Executed")
                    logging.info("Entering State: UPDATING")
                else:
                    current_waypoint = waypoints.pop(0)
                    navigation_graph.move_root(current_waypoint)
                    current_state = RobotState.PLANNING
                continue

        elif current_state is RobotState.UPDATING:
            logging.info("Current total distance: %.3f m", total_distance)
            spin_and_update(env, occupancy_map)
            refine_map(occupancy_map)
            save_nav_map(nav_dir, occupancy_map, navigation_graph)
            with open(exploration_stats_file, "a") as csvfile:
                csvwriter = csv.writer(csvfile)
                entry = [total_distance, occupancy_map.explored_space()]
                csvwriter.writerow(entry)

            current_state = RobotState.PLANNING
            logging.info("Entering State: PLANNING")

        elif current_state is RobotState.END:
            env.step(None)
            save_map(
                dir_path,
                robot_pos,
                occupancy_map,
                current_plan=[],
                frontier_line=[],
            )
            save_map_combo(
                combo_dir,
                robot_pos,
                occupancy_map,
                navigation_graph,
                current_plan=[],
                frontier_line=[],
            )
            save_nav_map(nav_dir, occupancy_map, navigation_graph)
            logging.info("Final total distance: %.3f m", total_distance)
            robot_pos = env.robots[0].get_position()[:2]

            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        choices=["beechwood", "wainscott", "ihlen", "benevolence0", "benevolence2"],
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=["random", "euclid", "simple", "visible", "info", "fusion"],
    )
    args = parser.parse_args()
    if args.method == "random":
        m = FrontierSelectionMethod.RANDOM
    if args.method == "euclid":
        m = FrontierSelectionMethod.CLOSEST_EUCLID
    if args.method == "visible":
        m = FrontierSelectionMethod.CLOSEST_GRAPH_VISIBLE
    if args.method == "simple":
        m = FrontierSelectionMethod.CLOSEST_GRAPH_SIMPLE
    if args.method == "info":
        m = FrontierSelectionMethod.BESTINFO
    if args.method == "fusion":
        m = FrontierSelectionMethod.FUSION

    dir_path = initiate_logging(
        "exploration.log", args.config + "/" + args.method + "/navgraph"
    )
    main(dir_path, args.config, m)
