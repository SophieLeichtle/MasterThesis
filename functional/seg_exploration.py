import logging
from random import sample
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from enum import IntEnum

from scipy.ndimage import binary_erosion

from soph import configs_path
from soph.environments.custom_env import CustomEnv
from soph.utils.occupancy_grid import OccupancyGrid2D

from soph.utils.motion_planning import plan_with_frontiers, teleport, plan_detection_frontier
from soph.utils.utils import fit_detections_to_point, check_detections_for_viewpoints
from yolo_utils import create_model, get_predictions, prepare_image, save_seg_image, get_detection

class RobotState(IntEnum):
    INIT = 0 # unused for now
    PLANNING = 1
    MOVING = 2
    UPDATING = 3
    END = 4

def main():
    """
    Create an igibson environment. 
    The robot tries to perform a simple frontiers based exploration of the environment.
    """

    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    config_filename = os.path.join(configs_path, "seg_explore.yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Create Environment
    env = CustomEnv(config_file=config_data, mode="gui_interactive")
    env.reset()

    # Create Map
    map = OccupancyGrid2D(half_size=350)
    
    model, device, names, colors = create_model("experiments/yolov7.pt", 640, False)

    # Initial Map Update
    state = env.get_state()
    robot_pos = env.robots[0].get_position()[:2]
    robot_theta = env.robots[0].get_rpy()[2]
    map.update_with_grid(occupancy_grid=state["occupancy_grid"], position=robot_pos, theta=robot_theta)
    
    current_state = RobotState.INIT
    # Init done outside loop for now
    for i in range(11):
        new_robot_theta = robot_theta + 0.2 * np.pi
        plan = [robot_pos[0], robot_pos[1], new_robot_theta]
        teleport(env, plan)

        state = env.get_state()
        robot_pos = env.robots[0].get_position()[:2]
        robot_theta = env.robots[0].get_rpy()[2]
        
        map.update_with_grid(occupancy_grid=state["occupancy_grid"], position=robot_pos, theta=robot_theta)
        
        #Sample points from depth sensor to accompany lidar occupancy grid
        depth = state["depth"]
        map.update_from_depth(env, depth)

    current_state = RobotState.PLANNING


    detected = False
    newest_detection = None
    detections = []

    while True:
        
        
        if current_state is RobotState.PLANNING:
            env.step(None)
            if not detected:
                current_plan = plan_with_frontiers(env, map)
            else: 
                print(detections)
                if len(detections) > 1 and check_detections_for_viewpoints(detections):
                    point = fit_detections_to_point(detections=detections)
                    point_in_map = map.m_to_px(point)
                    if map.grid[int(point_in_map[0]), int(point_in_map[1])]:
                        print(point_in_map)
                        current_state = RobotState.END
                        print("move to end state")
                        continue
                current_plan = plan_detection_frontier(env, map, detections[-1])
            if current_plan is not None:
                current_state = RobotState.MOVING

        elif current_state is RobotState.MOVING:
            current_point = current_plan.pop(0)
            teleport(env, current_point)
            if len(current_plan) == 0:
                current_plan = None
                current_state = RobotState.UPDATING
                continue
            if detected: continue

            detection = get_detection(env, model, device, names, colors, "chair", True)
            if detection is not None:
                detected = True
                detections.append(detection)
                current_plan = None
                current_state = RobotState.UPDATING
            

        elif current_state is RobotState.UPDATING:
            state = env.get_state()
            robot_pos = env.robots[0].get_position()[:2]
            robot_theta = env.robots[0].get_rpy()[2]
            map.update_with_grid(occupancy_grid=state["occupancy_grid"], position=robot_pos, theta=robot_theta)
            
            #Sample points from depth sensor to accompany lidar occupancy grid
            depth = state["depth"]
            map.update_from_depth(env, depth)
            current_state = RobotState.PLANNING

            detection = get_detection(env, model, device, names, colors, "chair", True)
            if detection is not None:
                detected = True
                detections.append(detection)


        elif current_state is RobotState.END:
            env.step(None)
            




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()