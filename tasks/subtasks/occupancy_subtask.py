from lib2to3.pytree import Base
import numpy as np

from tasks.subtasks.subtask_base import BaseSubtask

class OccupancySubtask(BaseSubtask):
    """
    Object Detection Task
    The goal is to detect an object with the robot camera
    """

    def __init__(self, env):
        super(OccupancySubtask, self).__init__(env)

        self.initial_pos = np.array(self.config.get("initial_pos", [0, 0, 0]))
        self.initial_orn = np.array(self.config.get("initial_orn", [0, 0, 0]))

        self.map_size = 250

        self.rot_dir = self.config.get("rot_dir", "clockwise")

    def reset_variables(self, env):
        self.occupancy_grid = np.zeros((self.map_size*2 + 1, self.map_size * 2 + 1))
        self.occupancy_grid.fill(0.5)


    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        task_obs = self.occupancy_grid

        return task_obs

    def step(self, env):
        """
        Perform task-specific step: step visualization and aggregate path length

        :param env: environment instance
        """
        new_robot_pos = env.robots[0].get_position()[:2]
        new_robot_theta = env.robots[0].get_rpy()[2]

        state = env.get_state()
        occupancy = state["occupancy_grid"]
        center = [self.map_size +1, self.map_size + 1]
        robot_pos_in_map = (new_robot_pos / 5.0 * 128 + center).astype(np.int32)
        c, s = np.cos(new_robot_theta), np.sin(new_robot_theta)
        R = np.array(((c,-s), (s, c)))

        for x in range(0,127):
            for y in range(0,127):
                #if y < 64 and np.arctan(np.abs((x-64)/(y-64))) < np.pi / 3:
                if occupancy[x][y] == 0.5:
                    continue
                point = (robot_pos_in_map + np.dot(R, [x-64,y-64])).astype(np.int32)
                self.occupancy_grid[point[0]][point[1]] = occupancy[x][y]
        self.robot_pos = new_robot_pos