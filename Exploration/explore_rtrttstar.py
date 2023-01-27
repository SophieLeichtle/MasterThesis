import os
import logging
from enum import IntEnum
import yaml
import numpy as np
import time

from soph import configs_path
from soph.environments.custom_env import CustomEnv
from soph.occupancy_grid.occupancy_grid import OccupancyGrid2D
from soph.occupancy_grid.occupancy_utils import spin_and_update, refine_map

from soph.planning.rt_rrt_star.rt_rrt_star import RTRRTstar
from soph.planning.rt_rrt_star.rt_rrt_star_planning import (
    next_goal,
    FrontierSelectionMethod,
)

from soph.planning.motion_planning import (
    teleport,
)
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


def main(dir_path):
    """
    Create an igibson environment.
    The robot tries to perform a simple frontiers based exploration of the environment.
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    config_filename = os.path.join(configs_path, "seg_explore copy.yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Create Environment
    logging.info("Creating Environment")
    env = CustomEnv(config_file=config_data, mode="gui_interactive")
    env.reset()

    # Create Map
    occupancy_map = OccupancyGrid2D(half_size=350)

    logging.info("Entering State: INIT")
    current_state = RobotState.INIT

    # Initial Map Update
    spin_and_update(env, occupancy_map)

    robot_pos = env.robots[0].get_position()[:2]
    rt_rrt_star = RTRRTstar(robot_pos)
    rt_rrt_star.initiate(occupancy_map)

    current_state = RobotState.PLANNING

    current_frontier = []
    current_plan = None

    logging.info("Entering State: PLANNING")

    planning_attempts = 0
    max_planning_attempts = 10

    iters = 0
    file_name = os.path.join(dir_path, f"{iters:05d}.png")
    save_map_rt_rrt_star(file_name, robot_pos, occupancy_map, rt_rrt_star)
    iters += 1

    # detailed_iters = 0
    # os.makedirs(os.path.join(dir_path, "detailed"))
    # file_name = os.path.join(dir_path, "detailed", f"{detailed_iters:05d}.png")
    # save_map_rt_rrt_star_detailed(file_name, occupancy_map, rt_rrt_star)
    # detailed_iters += 1

    total_distance = 0

    start_time = time.process_time()
    while True:
        if time.process_time() - start_time > 1:
            start_time = time.process_time()
            robot_pos = env.robots[0].get_position()[:2]
            file_name = os.path.join(dir_path, f"{iters:05d}.png")
            save_map_rt_rrt_star(
                file_name, robot_pos, occupancy_map, rt_rrt_star, None, current_frontier
            )

            iters += 1

        if current_state is RobotState.PLANNING:

            # env.step(None)
            robot_pos = env.robots[0].get_position()[:2]
            robot_theta = env.robots[0].get_rpy()[2]

            new_goal = None
            if len(current_frontier) == 0:
                planning_attempts += 1
                goal, frontier = next_goal(
                    env,
                    occupancy_map,
                    rt_rrt_star,
                    FrontierSelectionMethod.CLOSEST_GRAPH_VISIBLE,
                    True,
                )
                if goal is not None:
                    new_goal = goal[:2]
                    current_frontier = frontier
                    planning_attempts = 0
                else:
                    if planning_attempts == max_planning_attempts:
                        current_state = RobotState.END
                    continue
            current_plan, plan_completed = rt_rrt_star.nextIter(
                robot_pos, robot_theta, occupancy_map, new_goal
            )
            # file_name = os.path.join(dir_path, "detailed", f"{detailed_iters:05d}.png")
            # save_map_rt_rrt_star_detailed(file_name, occupancy_map, rt_rrt_star)
            # detailed_iters += 1

            if current_plan is None:
                if plan_completed:
                    current_frontier = []
                    current_state = RobotState.UPDATING

            else:
                current_state = RobotState.MOVING

        elif current_state is RobotState.MOVING:
            robot_pos = env.robots[0].get_position()[:2]
            total_distance += np.linalg.norm(robot_pos - current_plan[:2])
            teleport(env, current_plan)
            current_state = RobotState.PLANNING

        elif current_state is RobotState.UPDATING:
            logging.info("Current total distance: %.3f m", total_distance)
            spin_and_update(env, occupancy_map)
            refine_map(occupancy_map)
            current_state = RobotState.PLANNING
            logging.info("Entering State: PLANNING")

        elif current_state is RobotState.END:
            robot_pos = env.robots[0].get_position()[:2]
            file_name = os.path.join(dir_path, f"{iters:05d}.png")
            save_map_rt_rrt_star(
                file_name, robot_pos, occupancy_map, rt_rrt_star, None, current_frontier
            )
            break


if __name__ == "__main__":
    dir_path = initiate_logging("inseg_exploration.log")
    main(dir_path)
