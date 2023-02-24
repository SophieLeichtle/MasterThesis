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

from soph.planning.rt_rrt_star.rt_rrt_star import RTRRTstar
from soph.planning.rt_rrt_star.rt_rrt_star_planning import (
    next_goal,
)

from soph.planning.motion_planning import teleport, FrontierSelectionMethod
from soph.utils.logging_utils import (
    initiate_logging,
    save_map_rt_rrt_star,
    save_map_rt_rrt_star_detailed,
)

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

    # Create Environment
    logging.info("Creating Environment")
    env = CustomEnv(config_file=config_data, mode="gui_interactive")
    env.reset()

    logging.info("Frontier Selection Method: %s", frontier_method.name)

    # Create Map
    occupancy_map = OccupancyGrid2D(half_size=350)

    logging.info("Entering State: INIT")
    current_state = RobotState.INIT

    init_pos = initial_position(config)
    env.land(env.robots[0], init_pos, env.robots[0].get_rpy())

    # Create Map
    occupancy_map = OccupancyGrid2D(half_size=350)

    # Initial Map Update
    spin_and_update(env, occupancy_map)

    robot_pos = env.robots[0].get_position()[:2]
    rt_rrt_star = RTRRTstar(robot_pos)
    rt_rrt_star.initiate(occupancy_map)

    current_state = RobotState.PLANNING

    current_frontier = []
    current_plan = None
    current_goal = None

    logging.info("Entering State: PLANNING")

    planning_attempts = 0
    max_planning_attempts = 30

    rtt_iters = 0
    spin_iters = 100
    max_rtt_iters = 1000

    os.makedirs(os.path.join(dir_path, "images"))

    iters = 0
    file_name = os.path.join(dir_path, "images", f"{iters:05d}.png")
    save_map_rt_rrt_star(file_name, robot_pos, occupancy_map, rt_rrt_star)
    iters += 1

    # detailed_iters = 0
    # os.makedirs(os.path.join(dir_path, "detailed"))
    # file_name = os.path.join(dir_path, "detailed", f"{detailed_iters:05d}.png")
    # save_map_rt_rrt_star_detailed(file_name, occupancy_map, rt_rrt_star)
    # detailed_iters += 1

    total_distance = 0
    exploration_stats_file = os.path.join(dir_path, "exploration_stats.csv")
    planning_stats_file = os.path.join(dir_path, "planning_stats.csv")

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

    frames = 0
    while True:
        # cap at 30 minutes
        if time.process_time() - start_time > 1800:
            current_state = RobotState.END
        frames += 1
        if frames >= 30:
            frames = 0
            file_name = os.path.join(dir_path, "images", f"{iters:05d}.png")
            save_map_rt_rrt_star(
                file_name,
                robot_pos,
                occupancy_map,
                rt_rrt_star,
                None,
                current_frontier,
            )

            iters += 1

        if current_state is RobotState.PLANNING:

            # env.step(None)
            robot_pos = env.robots[0].get_position()[:2]
            robot_theta = env.robots[0].get_rpy()[2]

            new_goal = None
            if current_goal is None:
                logging.info("Making new plan")
                planning_attempts += 1
                t = time.process_time()
                goal, frontier = next_goal(
                    env,
                    occupancy_map,
                    rt_rrt_star,
                    frontier_method,
                    frontier_method == FrontierSelectionMethod.CLOSEST_GRAPH_VISIBLE,
                )
                t = time.process_time() - t
                with open(planning_stats_file, "a") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    entry = ["frontier", t, goal is not None]
                    csvwriter.writerow(entry)

                if goal is not None:
                    logging.info("New goal at %.2f , %.2f", goal[0], goal[1])
                    new_goal = goal[:2]
                    current_goal = goal
                    current_frontier = frontier
                    planning_attempts = 0
                else:
                    logging.info("Planning attempt %i", planning_attempts)
                    if planning_attempts >= max_planning_attempts:
                        logging.info(
                            "Max Planning attempts reached: Entering State END"
                        )
                        current_state = RobotState.END
                        continue
            t = time.process_time()
            current_plan, plan_completed = rt_rrt_star.nextIter(
                robot_pos, robot_theta, occupancy_map, new_goal
            )
            t = time.process_time() - t
            with open(planning_stats_file, "a") as csvfile:
                csvwriter = csv.writer(csvfile)
                entry = ["iter", t, True]
                csvwriter.writerow(entry)
            # file_name = os.path.join(dir_path, "detailed", f"{detailed_iters:05d}.png")
            # save_map_rt_rrt_star_detailed(file_name, occupancy_map, rt_rrt_star)
            # detailed_iters += 1

            if current_plan is None:
                if plan_completed and current_goal is not None:
                    current_frontier = []
                    current_plan = current_goal
                    current_goal = None
                    current_state = RobotState.MOVING
                    rtt_iters = 0
                else:
                    rtt_iters += 1
                    if rtt_iters == spin_iters:
                        logging.info(
                            "RTT Iterations reached Spin Threshold: Updating Map Once"
                        )
                        current_state = RobotState.UPDATING
                    if rtt_iters >= max_rtt_iters:
                        logging.info("Max RTT Iterations reached: Entering State END")
                        current_state = RobotState.END

            else:
                rtt_iters = 0
                current_state = RobotState.MOVING

        elif current_state is RobotState.MOVING:
            robot_pos = env.robots[0].get_position()[:2]
            total_distance += np.linalg.norm(robot_pos - current_plan[:2])
            teleport(env, current_plan)
            if current_goal is None:
                current_state = RobotState.UPDATING
            else:
                current_state = RobotState.PLANNING

        elif current_state is RobotState.UPDATING:
            logging.info("Current total distance: %.3f m", total_distance)
            spin_and_update(env, occupancy_map)
            refine_map(occupancy_map)
            rt_rrt_star.map = occupancy_map
            with open(exploration_stats_file, "a") as csvfile:
                csvwriter = csv.writer(csvfile)
                entry = [total_distance, occupancy_map.explored_space()]
                csvwriter.writerow(entry)

            current_state = RobotState.PLANNING
            logging.info("Entering State: PLANNING")

        elif current_state is RobotState.END:
            robot_pos = env.robots[0].get_position()[:2]
            file_name = os.path.join(dir_path, "final.png")
            save_map_rt_rrt_star(
                file_name, robot_pos, occupancy_map, rt_rrt_star, None, current_frontier
            )
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
        "exploration.log", args.config + "/" + args.method + "/rtrrtstar"
    )
    main(dir_path, args.config, m)
