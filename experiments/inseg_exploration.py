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
from yolo_mask_utils import create_model, get_detection

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
    config_filename = os.path.join(configs_path, "seg_explore copy.yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Create Environment
    env = CustomEnv(config_file=config_data, mode="gui_interactive")
    env.reset()

    # Create Map
    map = OccupancyGrid2D(half_size=350)
    
    model, device, hyp = create_model()

    # Initial Map Update
    state = env.get_state()
    robot_pos = env.robots[0].get_position()[:2]
    robot_theta = env.robots[0].get_rpy()[2]
    map.update_with_grid(occupancy_grid=state["occupancy_grid"], position=robot_pos, theta=robot_theta)
    current_state = RobotState.INIT
    # Init done outside loop for now
    for i in range(10):
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
    print("Starting Planning")
    while True:
        
        
        if current_state is RobotState.PLANNING:
            env.step(None)
            if not detected:
                current_plan = plan_with_frontiers(env, map)
            else: 
                if len(detections) > 1 and check_detections_for_viewpoints(detections):
                    point = fit_detections_to_point(detections=detections)
                    point_in_map = map.m_to_px(point)
                    if map.grid[int(point_in_map[0]), int(point_in_map[1])] != 0.5:
                        print(point_in_map)
                        current_state = RobotState.END
                        print("move to end state")
                        continue
                current_plan = plan_detection_frontier(env, map, detections[-1])
            if current_plan is not None:
                current_state = RobotState.MOVING
                print("Starting Moving")
            else:
                plt.figure()
                plt.imshow(map.grid)
                plt.savefig("map.jpg")
                plt.close()
                input("Enter")
                

        elif current_state is RobotState.MOVING:
            current_point = current_plan.pop(0)
            teleport(env, current_point)
            if len(current_plan) == 0:
                current_plan = None
                current_state = RobotState.UPDATING
                print("Starting Updating")
                continue
            if detected: continue

            detection, mask = get_detection(env, model, device, hyp, "chair", True)
            if detection is not None:
                detected = True
                detections.append(detection)
                current_plan = None
                current_state = RobotState.UPDATING
                print("Starting Updating")
            

        elif current_state is RobotState.UPDATING:
            state = env.get_state()
            robot_pos = env.robots[0].get_position()[:2]
            robot_theta = env.robots[0].get_rpy()[2]
            map.update_with_grid(occupancy_grid=state["occupancy_grid"], position=robot_pos, theta=robot_theta)
            
            #Sample points from depth sensor to accompany lidar occupancy grid
            depth = state["depth"]
            map.update_from_depth(env, depth)
            

            detection, mask = get_detection(env, model, device, hyp, "chair", True)
            if detection is not None:
                detected = True
                detections.append(detection)
                masked_depth = depth[:,:,0] * mask
                if masked_depth.max() > 0:
                    current_state = RobotState.END
                    print("move to end state")
                    continue
            
            current_state = RobotState.PLANNING
            print("Starting Planning")


        elif current_state is RobotState.END:
            env.step(None)




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()