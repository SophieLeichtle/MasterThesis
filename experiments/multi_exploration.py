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
from soph.utils.occupancy_grid import OccupancyGrid2D

from soph.utils.motion_planning import plan_with_frontiers, teleport, get_poi, plan_base_motion, plan_frontier_with_poi
from soph.utils.logging_utils import save_map, initiate_logging
from soph.utils.utils import bbox, pixel_to_point

from soph.utils.detection_tool import DetectionTool, DefinitiveDetection

from yolo_mask_utils import create_model, get_detections
class RobotState(IntEnum):
    INIT = 0 # unused for now
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
    map = OccupancyGrid2D(half_size=350)
    
    logging.info("Creating Model")
    model, device, hyp = create_model()

    logging.info("Entering State: INIT")

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

    

    detection_tool = DetectionTool()

    save_map(log_dir, robot_pos, map, detection_tool)

    logging.info("Entering State: PLANNING")
    
    planning_attempts = 0
    max_planning_attempts = 10
    while True:
        if current_state is RobotState.PLANNING:
            planning_attempts += 1
            env.step(None)
            if len(detection_tool.pois) == 0:
                logging.info("Planning with Frontiers")
                current_plan = plan_with_frontiers(env, map)
            else: 
                logging.info("Planning with POIs")
                robot_pos = env.robots[0].get_position()[:2]
                closest_poi = detection_tool.closest_poi(robot_pos)
                if map.check_if_free(closest_poi[:2]):
                    current_plan = plan_base_motion(env.robots[0], closest_poi, map)
                else:
                    current_plan = plan_frontier_with_poi(env, map, closest_poi)
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
            detection_tool.remove_close_pois(current_point)
            if len(current_plan) == 0:
                current_plan = None
                current_state = RobotState.UPDATING
                logging.info("Current Plan Executed")
                logging.info("Entering State: UPDATING")
                continue

            detections, masks = get_detections(env, model, device, hyp, queried_semantic, True)
            if detections is not None:
                state = env.get_state()
                depth = state["depth"]
                for detection, mask in zip(detections, masks):
                    masked_depth = depth[:,:,0] * mask
                    if np.count_nonzero(masked_depth) > 50:
                        rmin, rmax, cmin, cmax = bbox(masked_depth)
                        points = []
                        for r in range(rmin, rmax+1):
                            for c in range(cmin, cmax+1):
                                d = masked_depth[r,c]
                                if d == 0: continue
                                point = pixel_to_point(env, r, c, d)
                                if point[2] > 0.05: points.append(point)
                        
                        new_detection = detection_tool.register_definitive_detection(points)
                        if new_detection is not None:
                            logging.info("New Detection Located at " + f'{new_detection.position[0]:.2f}, {new_detection.position[1]:.2f}')
                            input("enter")
                    else:
                        poi = get_poi(detection)
                        new = detection_tool.register_new_poi(poi)
                        if new:
                            logging.info("Object Detected: New POI added")
            

        elif current_state is RobotState.UPDATING:
            state = env.get_state()
            robot_pos = env.robots[0].get_position()[:2]
            robot_theta = env.robots[0].get_rpy()[2]
            map.update_with_grid(occupancy_grid=state["occupancy_grid"], position=robot_pos, theta=robot_theta)
            
            #Sample points from depth sensor to accompany lidar occupancy grid
            depth = state["depth"]
            map.update_from_depth(env, depth)
            
            save_map(log_dir, robot_pos, map, detection_tool)

            
            current_state = RobotState.PLANNING
            logging.info("Entering State: PLANNING")


        elif current_state is RobotState.END:
            env.step(None)




if __name__ == "__main__":
    dir_path = initiate_logging("inseg_exploration.log")
    main(dir_path)