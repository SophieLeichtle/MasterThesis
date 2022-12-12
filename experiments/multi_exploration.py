import os
import logging
from enum import IntEnum
import yaml
import numpy as np

from yolo_mask_utils import create_model, get_detections

from soph import configs_path
from soph.environments.custom_env import CustomEnv
from soph.utils.occupancy_grid import OccupancyGrid2D

from soph.utils.motion_planning import (
    teleport,
    get_poi,
    plan_with_poi,
    plan_base_motion,
    frontier_plan_with_nav,
    sample_plan_poi,
)
from soph.utils.logging_utils import (
    save_map,
    initiate_logging,
    save_nav_map,
    save_map_combo,
)
from soph.utils.utils import bbox, pixel_to_point
from soph.utils.nav_graph import NavGraph


from soph.utils.detection_tool import DetectionTool


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

    queried_semantic = "chair"

    # Create Environment
    logging.info("Creating Environment")
    env = CustomEnv(config_file=config_data, mode="gui_interactive")
    env.reset()

    # Create Map
    occupancy_map = OccupancyGrid2D(half_size=350)

    logging.info("Creating Model")
    model, device, hyp = create_model()

    logging.info("Entering State: INIT")

    # Initial Map Update
    state = env.get_state()
    robot_pos = env.robots[0].get_position()[:2]
    robot_theta = env.robots[0].get_rpy()[2]
    occupancy_map.update_with_grid(
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

        occupancy_map.update_with_grid(
            occupancy_grid=state["occupancy_grid"],
            position=robot_pos,
            theta=robot_theta,
        )

        # Sample points from depth sensor to accompany lidar occupancy grid
        depth = state["depth"]
        occupancy_map.update_from_depth(env, depth, 5000)

    navigation_graph = NavGraph(np.array(robot_pos))

    current_state = RobotState.PLANNING

    detection_tool = DetectionTool()

    save_map(log_dir, robot_pos, occupancy_map, detection_tool)

    logging.info("Entering State: PLANNING")

    verbose_planning = False

    planning_attempts = 0
    max_planning_attempts = 10

    waypoints = None
    current_frontier = None
    frontier_plan = None

    while True:
        if current_state is RobotState.PLANNING:
            planning_attempts += 1
            env.step(None)
            robot_pos = env.robots[0].get_position()[:2]

            if len(detection_tool.pois) == 0:
                if waypoints is None:
                    logging.info("Planning With Frontiers")
                    current_plan = None
                    waypoints, frontier_plan, current_frontier = frontier_plan_with_nav(
                        env, occupancy_map, navigation_graph
                    )

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
                        current_plan = sample_plan_poi(
                            env,
                            occupancy_map,
                            np.array([next_point[0], next_point[1], theta]),
                        )

            else:
                logging.info("Planning with POIs")
                closest_poi = detection_tool.closest_poi(robot_pos)
                current_plan, current_frontier = plan_with_poi(
                    env, occupancy_map, closest_poi, verbose=verbose_planning
                )
                waypoints = None
            if current_plan is not None:
                planning_attempts = 0
                current_state = RobotState.MOVING
                logging.info("Entering State: MOVING")
                save_map(
                    log_dir,
                    robot_pos,
                    occupancy_map,
                    detection_tool,
                    current_plan,
                    current_frontier,
                )
                save_map_combo(
                    log_dir,
                    robot_pos,
                    occupancy_map,
                    detection_tool,
                    navigation_graph,
                    current_plan,
                    current_frontier,
                )
            else:
                if planning_attempts <= max_planning_attempts:
                    logging.warning("No plan found. Attempting Planning again.")
                    logging.info("Turning on Verbose Planning")
                    verbose_planning = True
                else:
                    logging.warning("Max Planning Attempts reached")
                    logging.info("Entering State: End")
                    save_map(
                        log_dir,
                        robot_pos,
                        occupancy_map,
                        detection_tool,
                        frontier_line=current_frontier,
                    )
                    save_nav_map(log_dir, occupancy_map, navigation_graph)
                    save_map_combo(
                        log_dir,
                        robot_pos,
                        occupancy_map,
                        detection_tool,
                        navigation_graph,
                    )

                    current_state = RobotState.END

        elif current_state is RobotState.MOVING:
            current_point = current_plan.pop(0)
            teleport(env, current_point)
            detection_tool.remove_close_pois(current_point)

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
                        for row in range(rmin, rmax + 1):
                            for col in range(cmin, cmax + 1):
                                d = masked_depth[row, col]
                                if d == 0:
                                    continue
                                point = pixel_to_point(env, row, col, d)
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
                            current_plan = None
                            waypoints = None
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

        elif current_state is RobotState.UPDATING:
            env.step(None)

            state = env.get_state()
            robot_pos = env.robots[0].get_position()[:2]
            robot_theta = env.robots[0].get_rpy()[2]
            occupancy_map.update_with_grid(
                occupancy_grid=state["occupancy_grid"],
                position=robot_pos,
                theta=robot_theta,
            )

            # Sample points from depth sensor to accompany lidar occupancy grid
            depth = state["depth"]
            occupancy_map.update_from_depth(env, depth)

            for i in range(10):
                new_robot_theta = robot_theta + 0.2 * np.pi
                plan = [robot_pos[0], robot_pos[1], new_robot_theta]
                teleport(env, plan)

                state = env.get_state()
                robot_pos = env.robots[0].get_position()[:2]
                robot_theta = env.robots[0].get_rpy()[2]

                occupancy_map.update_with_grid(
                    occupancy_grid=state["occupancy_grid"],
                    position=robot_pos,
                    theta=robot_theta,
                )

                # Sample points from depth sensor to accompany lidar occupancy grid
                depth = state["depth"]
                occupancy_map.update_from_depth(env, depth, 5000)

            navigation_graph.update_with_robot_pos(robot_pos, occupancy_map)
            save_nav_map(log_dir, occupancy_map, navigation_graph)

            current_state = RobotState.PLANNING
            logging.info("Entering State: PLANNING")
            # input("enter")

        elif current_state is RobotState.END:
            env.step(None)


if __name__ == "__main__":
    dir_path = initiate_logging("inseg_exploration.log")
    main(dir_path)
