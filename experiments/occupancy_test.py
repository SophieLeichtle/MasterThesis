import logging
import os
from sys import platform
from turtle import pos
import matplotlib.pyplot as plt
import yaml
import numpy as np
import open3d as o3d
import cv2
from enum import IntEnum
import random


import igibson
from igibson.render.profiler import Profiler
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings

from soph.utils.occupancy_grid import OccupancyGrid2D
from soph.utils.simple_pi import SimplePI
from soph.environments.custom_env import CustomEnv
from soph.utils.utils import pixel_to_point

class RobotState(IntEnum):
    PLANNING = 0
    MOVING = 1
    END = 2

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


    rendering_settings = MeshRendererSettings(optimized=False)
    env = CustomEnv(config_file=config_data, rendering_settings=rendering_settings, mode="gui_interactive")


    max_iterations = 1 if not short_exec else 1
    for j in range(max_iterations):
        print("Resetting environment")
        env.reset()

        grid = OccupancyGrid2D()

        state, reward, done, info = env.step(None)

        rstate = RobotState.PLANNING
        current_plan = None
        pi_controller = SimplePI(.5, 0, env.action_timestep)

        action = None
        motion_planner = MotionPlanningWrapper(env, visualize_2d_result=False)

        while rstate is not RobotState.END:
            state, reward, done, info = env.step(action)
            new_robot_pos = env.robots[0].get_position()[:2]
            new_robot_theta = env.robots[0].get_rpy()[2]
            if rstate is RobotState.PLANNING:

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
                        
                #Update Grid
                grid.update_with_grid(occupancy_grid=state["occupancy_grid"], position=new_robot_pos, theta=new_robot_theta)
                plt.figure()
                plt.imshow(grid.grid)
                plt.show()

                grid.update_with_points(points)
                plt.figure()
                plt.imshow(grid.grid)
                plt.show()   

                iter = 0
                current_plan = None

                while(current_plan == None or current_plan == []):
                    if iter > 20:
                        print("max iterations reached")
                        rstate = RobotState.END
                        break
                    iter += 1
                    best_info = 0
                    best_point = None
                    best_theta = None

                    #Sample Points within Range
                    num_points = 100
                    ranges = np.random.uniform(0, 1.0, num_points)
                    angles = np.random.uniform(0, 2 * np.pi, num_points)
                    num_thetas = 10
                    thetas = np.random.uniform(-np.pi, np.pi, num_thetas)

                    #Find Point with most new info
                    for i in range(0,num_points):
                        unitv = np.array([np.cos(angles[i]), np.sin(angles[i])])
                        point = new_robot_pos + ranges[i] * unitv
                        if not grid.check_if_free(point, 0.35): continue
                        for y in range(0, num_thetas):
                            new_info = grid.check_new_information(point, thetas[y], 2.5, 1.5 * np.pi)
                            if new_info > best_info:
                                best_info = new_info
                                best_point = point
                                best_theta = thetas[y]
                    if best_info == 0: continue
                    plan = motion_planner.plan_base_motion([best_point[0], best_point[1], best_theta])
                    current_plan = plan
                else:
                    goal = np.array(current_plan[0])
                    robot_state = np.array([new_robot_pos[0], new_robot_pos[1], new_robot_theta])
                    pi_controller.reset_goal(goal, robot_state)
                
                    #TODO replace with real movement/control
                    motion_planner.dry_run_base_plan(current_plan)
                    current_plan = None
                    print(best_info)
                    print(best_point)
                    print(best_theta)
                    grid.check_new_information(best_point, best_theta, 2.5, 1.5*np.pi, True)
                    if best_info < 10:
                        rstate = RobotState.END

            if rstate is RobotState.MOVING:
                precision = 0.01
                goal = np.array(current_plan[0])
                robot_state = np.array([new_robot_pos[0], new_robot_pos[1], new_robot_theta])
                
                action = pi_controller.get_control(robot_state)
                print(action)
                print(goal - robot_state)
                if np.linalg.norm(action, ord=np.inf) < 0.001:
                    print("intermediate goal reached")
                    print(goal)
                    print(robot_state)
                    current_plan.pop(0)
                    if current_plan == []:
                        rstate = RobotState.END
                        continue
                    goal = np.array(current_plan[0])
                    pi_controller.reset_goal(goal, robot_state)
                state, reward, done, info = env.step(action)

        if rstate is RobotState.END:
            state, reward, done, info = env.step(None)
            new_robot_pos = env.robots[0].get_position()[:2]
            new_robot_theta = env.robots[0].get_rpy()[2]
            grid.update_with_grid(occupancy_grid=state["occupancy_grid"], position=new_robot_pos, theta=new_robot_theta)
            plt.figure()
            plt.imshow(grid.grid)
            plt.show()            
        
    env.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
