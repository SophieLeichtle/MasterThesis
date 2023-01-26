import logging
from random import sample
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from enum import IntEnum

from soph import configs_path
from soph.environments.custom_env import CustomEnv
from soph.occupancy_grid.occupancy_grid import OccupancyGrid2D

from soph.utils.motion_planning import (
    frontier_plan_bestinfo,
    teleport,
    frontier_plan_detection,
)
from soph.utils.utils import fit_detections_to_point, check_detections_for_viewpoints
from soph.utils.logging_utils import save_map, initiate_logging

from experiments.yolo_mask_utils import create_model, get_detection


class RobotState(IntEnum):
    INIT = 0  # unused for now
    PLANNING = 1
    MOVING = 2
    UPDATING = 3
    END = 4


def main(log_dir):
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
    map = OccupancyGrid2D(half_size=350)

    logging.info("Creating Model")
    model, device, hyp = create_model()

    logging.info("Entering State: INIT")

    # Initial Map Update
    state = env.get_state()
    robot_pos = env.robots[0].get_position()[:2]
    robot_theta = env.robots[0].get_rpy()[2]
    map.update_with_grid(
        occupancy_grid=state["occupancy_grid"], position=robot_pos, theta=robot_theta
    )

    current_state = RobotState.INIT
    # Init done outside loop for now
    for i in range(10):
        new_robot_theta = robot_theta + 0.2 * np.pi
        plan = [robot_pos[0], robot_pos[1], new_robot_theta]
        teleport(env, plan)

        state = env.get_state()
        robot_pos = env.robots[0].get_position()[:2]
        robot_theta = env.robots[0].get_rpy()[2]

        map.update_with_grid(
            occupancy_grid=state["occupancy_grid"],
            position=robot_pos,
            theta=robot_theta,
        )

        # Sample points from depth sensor to accompany lidar occupancy grid
        depth = state["depth"]
        map.update_from_depth(env, depth)

    current_state = RobotState.PLANNING

    save_map(log_dir, map.grid)

    detected = False
    detections = []
    logging.info("Entering State: PLANNING")

    planning_attempts = 0
    max_planning_attempts = 10
    while True:
        if current_state is RobotState.PLANNING:
            planning_attempts += 1
            env.step(None)
            if not detected:
                current_plan = frontier_plan_bestinfo(env, map)
            else:
                if len(detections) > 1 and check_detections_for_viewpoints(detections):
                    point = fit_detections_to_point(detections=detections)
                    point_in_map = map.m_to_px(point)
                    if map.grid[int(point_in_map[0]), int(point_in_map[1])] != 0.5:
                        current_state = RobotState.END
                        logging.info("Arrived at Goal")
                        logging.info("Simulation time: " + f"{sim_time}s")
                        logging.info("Entering State: END")
                        continue
                current_plan = frontier_plan_detection(env, map, detections[-1])
            if current_plan is not None:
                current_state = RobotState.MOVING
                logging.info("Entering State: MOVING")
                planning_attempts = 0
            else:
                if planning_attempts <= max_planning_attempts:
                    logging.warning("No plan found. Attempting Planning again.")
                else:
                    logging.warning("Max Planning Attempts reached")
                    logging.info("Entering State: End")
                    current_state = RobotState.END

        elif current_state is RobotState.MOVING:
            current_point = current_plan.pop(0)
            teleport(env, current_point)
            if len(current_plan) == 0:
                current_plan = None
                current_state = RobotState.UPDATING
                logging.info("Current Plan Executed")
                logging.info("Entering State: UPDATING")
                continue
            if detected:
                continue

            detection, mask = get_detection(
                env, model, device, hyp, "dining table", True
            )
            if detection is not None:
                detected = True
                detections.append(detection)
                current_plan = None
                current_state = RobotState.UPDATING
                logging.info("First Detection Made")
                logging.info("Entering State: UPDATING")

        elif current_state is RobotState.UPDATING:
            state = env.get_state()
            robot_pos = env.robots[0].get_position()[:2]
            robot_theta = env.robots[0].get_rpy()[2]
            map.update_with_grid(
                occupancy_grid=state["occupancy_grid"],
                position=robot_pos,
                theta=robot_theta,
            )

            # Sample points from depth sensor to accompany lidar occupancy grid
            depth = state["depth"]
            map.update_from_depth(env, depth)

            save_map(log_dir, map.grid)

            detection, mask = get_detection(
                env, model, device, hyp, "dining table", True
            )
            if detection is not None:
                detected = True
                detections.append(detection)
                masked_depth = depth[:, :, 0] * mask
                if masked_depth.max() > 0:
                    current_state = RobotState.END
                    sim_time = env.simulation_time()
                    logging.info("Arrived at Goal")
                    logging.info("Simulation time: " + f"{sim_time}")
                    logging.info("Entering State: END")
                    continue

            current_state = RobotState.PLANNING
            logging.info("Entering State: PLANNING")

        elif current_state is RobotState.END:
            env.step(None)


if __name__ == "__main__":
    dir_path = initiate_logging("inseg_exploration.log")
    main(dir_path)
