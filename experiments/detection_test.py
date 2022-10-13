import logging
import os
from sys import platform
import matplotlib.pyplot as plt
import yaml
import numpy as np
import open3d as o3d
import cv2
import octomap


import igibson
from igibson.render.profiler import Profiler

from src.environments.custom_env import CustomEnv

def main(selection="user", headless=False, short_exec=False):
    """
    Creates an iGibson environment from a config file with a turtlebot in Rs_int (interactive).
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    config_filename = "config/test_config.yaml"
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Improving visuals in the example (optional)
    config_data["enable_shadow"] = True
    config_data["enable_pbr"] = True

    env = CustomEnv(config_file=config_data, mode="gui_interactive")

    pointcloud = None

    max_iterations = 1 if not short_exec else 1
    for j in range(max_iterations):
        print("Resetting environment")
        env.reset()
        for i in range(100):
            with Profiler("Environment action step"):
                action = [0,-.1]#env.action_space.sample()
                state, reward, done, info = env.step(action)
                #if done:
                    #print("Episode finished after {} timesteps".format(i + 1))
                    #break
        
        pointcloud = info["scanning"]
    env.close()
    occupied, empty = pointcloud.extractPointCloud()
    print(occupied.shape)

    #plt.figure()
    #plt.imshow(info["occupancy"])
    #plt.show()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(occupied)
    o3d.io.write_point_cloud("testocto.ply", pcd)
    #cv2.destroyAllWindows()
    #o3d.visualization.draw_geometries([pointcloud], window_name='newWindow')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
