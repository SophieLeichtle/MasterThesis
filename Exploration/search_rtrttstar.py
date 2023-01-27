import os
import logging
from enum import IntEnum
import yaml
import numpy as np
import cv2
import time

from ..yolo.yolo_mask_utils import create_model, get_detections

from soph import configs_path
from soph.environments.custom_env import CustomEnv
from soph.occupancy_grid.occupancy_grid import OccupancyGrid2D
from soph.occupancy_grid.occupancy_utils import (
    spin_and_update,
    update_grid_with_scan,
    refine_map,
)

from soph.planning.rt_rrt_star.rt_rrt_star import RTRRTstar
from soph.planning.rt_rrt_star.rt_rrt_star_planning import (
    closest_frontier,
    goal_from_poi,
)

from soph.planning.motion_planning import (
    teleport,
    get_poi,
)
from soph.utils.logging_utils import (
    initiate_logging,
    save_map_rt_rrt_star,
)
from soph.utils.utils import bbox, px_to_3d, openglf_to_wf
from soph.planning.nav_graph.nav_graph import NavGraph


from soph.utils.detection_tool import DetectionTool
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

    queried_semantic = "chair"

    # Create Environment
    logging.info("Creating Environment")
    env = CustomEnv(config_file=config_data, mode="gui_interactive")
    env.reset()

    # Create Map
    occupancy_map = OccupancyGrid2D(half_size=350)

    logging.info("Creating Model")
    model, device, hyp = create_model()

    detection_tool = DetectionTool()

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
    current_goal = None

    logging.info("Entering State: PLANNING")

    planning_attempts = 0
    max_planning_attempts = 50

    iters = 0
    file_name = os.path.join(dir_path, f"{iters:05d}.png")
    save_map_rt_rrt_star(file_name, robot_pos, occupancy_map, rt_rrt_star)
    iters += 1

    total_distance = 0

    start_time = time.process_time()

    frames = 0
    while True:
        frames += 1
        if frames >= 30:
            frames = 0
            start_time = time.process_time()

            file_name = os.path.join(dir_path, f"{iters:05d}.png")
            save_map_rt_rrt_star(
                file_name,
                robot_pos,
                occupancy_map,
                rt_rrt_star,
                detection_tool,
                current_frontier,
            )

            iters += 1

        if current_state is RobotState.PLANNING:

            # env.step(None)
            robot_pos = env.robots[0].get_position()[:2]
            robot_theta = env.robots[0].get_rpy()[2]

            new_goal = None
            if current_goal is None:
                planning_attempts += 1
                if len(detection_tool.pois) == 0:

                    goal, frontier = closest_frontier(occupancy_map, rt_rrt_star)

                else:
                    closest_poi = detection_tool.closest_poi(robot_pos)
                    goal, frontier = goal_from_poi(
                        closest_poi, occupancy_map, rt_rrt_star
                    )

                if goal is not None:
                    new_goal = goal[:2]
                    current_goal = goal
                    current_frontier = frontier
                    planning_attempts = 0
                else:
                    if planning_attempts == max_planning_attempts:
                        current_state = RobotState.END
                    continue

            current_plan, plan_completed = rt_rrt_star.nextIter(
                robot_pos, robot_theta, occupancy_map, new_goal
            )

            if current_plan is None:
                if plan_completed:
                    current_frontier = []
                    current_plan = current_goal
                    current_goal = None
                    current_state = RobotState.MOVING

            else:
                current_state = RobotState.MOVING

        elif current_state is RobotState.MOVING:
            robot_pos = env.robots[0].get_position()[:2]
            total_distance += np.linalg.norm(robot_pos - current_plan[:2])
            teleport(env, current_plan)
            detection_tool.remove_close_pois(current_plan)

            detections, masks = get_detections(
                env, model, device, hyp, queried_semantic, True
            )
            if detections is not None:
                state = env.get_state()
                depth = state["depth"]
                for detection, mask in zip(detections, masks):
                    masked_depth = depth[:, :, 0] * mask
                    if np.count_nonzero(masked_depth) > 50:
                        rmin, rmax, cmin, cmax = bbox(masked_depth)
                        points = []
                        t_mat = openglf_to_wf(env.robots[0])
                        for row in range(rmin, rmax + 1):
                            for col in range(cmin, cmax + 1):
                                d = masked_depth[row, col]
                                if d == 0:
                                    continue
                                point = px_to_3d(
                                    row, col, d, t_mat, env.config["depth_high"]
                                )
                                if point[2] > 0.05:
                                    points.append(point)

                        new_detection = detection_tool.register_definitive_detection(
                            points
                        )
                        if new_detection is not None:
                            logging.info(
                                "New Detection Located at %.2f, %.2f",
                                new_detection.position[0],
                                new_detection.position[1],
                            )
                            current_goal = None
                            current_state = RobotState.UPDATING
                            logging.info("Current Plan Aborted")
                            logging.info("Entering State: UPDATING")
                    else:
                        poi = get_poi(detection)
                        new = detection_tool.register_new_poi(poi)
                        if new:
                            logging.info(
                                "Object Detected: New POI added at %.2f, %.2f",
                                poi[0],
                                poi[1],
                            )
                            update_grid_with_scan(env, occupancy_map)

            if current_goal is None or current_state == RobotState.UPDATING:
                current_state = RobotState.UPDATING
            else:
                current_state = RobotState.PLANNING

        elif current_state is RobotState.UPDATING:
            logging.info("Current total distance: %.3f m", total_distance)
            spin_and_update(env, occupancy_map)
            refine_map(occupancy_map)
            current_state = RobotState.PLANNING
            logging.info("Entering State: PLANNING")
            # input("enter")

        elif current_state is RobotState.END:
            logging.info("Entering State: END")
            logging.info("Final total distance: %.3f m", total_distance)
            robot_pos = env.robots[0].get_position()[:2]
            file_name = os.path.join(dir_path, f"{iters:05d}.png")
            save_map_rt_rrt_star(
                file_name,
                robot_pos,
                occupancy_map,
                rt_rrt_star,
                detection_tool,
                current_frontier,
            )
            break


if __name__ == "__main__":
    dir_path = initiate_logging("inseg_exploration.log")
    main(dir_path)
