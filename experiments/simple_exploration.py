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

from soph.utils.motion_planning import dry_run_base_plan, extract_frontiers, plan_base_motion, sample_around_frontier
from soph.utils.utils import pixel_to_point, bbox

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
    config_filename = os.path.join(configs_path, "simple_explore.yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Create Environment
    env = CustomEnv(config_file=config_data, mode="gui_interactive")
    env.reset()

    # Create Map
    map = OccupancyGrid2D()
    
    # Initial Map Update
    state = env.get_state()
    robot_pos = env.robots[0].get_position()[:2]
    robot_theta = env.robots[0].get_rpy()[2]
    map.update_with_grid(occupancy_grid=state["occupancy_grid"], position=robot_pos, theta=robot_theta)
    
    current_state = RobotState.INIT
    # Init done outside loop for now
    for i in range(11):
        new_robot_theta = robot_theta + 0.2 * np.pi
        plan = [[robot_pos[0], robot_pos[1], new_robot_theta]]
        dry_run_base_plan(env, plan)

        state = env.get_state()
        robot_pos = env.robots[0].get_position()[:2]
        robot_theta = env.robots[0].get_rpy()[2]
        
        map.update_with_grid(occupancy_grid=state["occupancy_grid"], position=robot_pos, theta=robot_theta)
        
        #Sample points from depth sensor to accompany lidar occupancy grid
        depth = state["depth"]
        samplesize = 1000
        rows = np.random.randint(depth.shape[0], size = samplesize)
        columns = np.random.randint(depth.shape[0], size = samplesize)
        points = []
        for it in range(0, samplesize):
            d = depth[rows[it], columns[it],0]
            if d == 0 : continue
            p = pixel_to_point(env, rows[it], columns[it], d)
            if p[2] > 0.05:
                points.append(p)
        map.update_with_points(points)

    current_state = RobotState.PLANNING

    while True:
        state = env.get_state()
        robot_pos = env.robots[0].get_position()[:2]
        robot_theta = env.robots[0].get_rpy()[2]
        if current_state is RobotState.PLANNING:
            best_info = 0
            best_sample = None
            current_plan = None

            frontier_lines = extract_frontiers(map.grid)
            for line in frontier_lines:
                if len(line) > 10:
                    samples = sample_around_frontier(line, map)
                    print(samples)
                    samples.sort(key=lambda x: np.linalg.norm([robot_pos[0] - x[0], robot_pos[1] - x[1], robot_theta - x[2]]))
                    for s in samples:
                        if not map.check_if_free(s, 0.35): continue
                        new_info = map.check_new_information(np.array([s[0],s[1]]), s[2], 2.5, 1.5*np.pi)
                        if new_info > best_info:
                            plan = plan_base_motion(env.robots[0], s, map)
                            if plan is None: continue
                            best_info = new_info
                            best_sample = s
                            current_plan = plan
                            break
    
            print(best_sample)
            if current_plan is not None:
                current_state = RobotState.MOVING
        elif current_state is RobotState.MOVING:
            dry_run_base_plan(env, current_plan)
            current_state = RobotState.UPDATING
        elif current_state is RobotState.UPDATING:
            map.update_with_grid(occupancy_grid=state["occupancy_grid"], position=robot_pos, theta=robot_theta)
            
            #Sample points from depth sensor to accompany lidar occupancy grid
            depth = state["depth"]
            samplesize = 1000
            rows = np.random.randint(depth.shape[0], size = samplesize)
            columns = np.random.randint(depth.shape[0], size = samplesize)
            points = []
            for it in range(0, samplesize):
                d = depth[rows[it], columns[it],0]
                if d == 0 : continue
                p = pixel_to_point(env, rows[it], columns[it], d)
                if p[2] > 0.05:
                    points.append(p)
            map.update_with_points(points)
            plt.figure()
            plt.imshow(map.grid)
            plt.show()
            current_state = RobotState.PLANNING
        elif current_state is RobotState.END:
            break




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()