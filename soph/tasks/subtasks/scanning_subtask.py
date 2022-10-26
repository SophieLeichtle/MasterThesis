from lib2to3.pytree import Base
import numpy as np
import matplotlib.pyplot as plt

import open3d as o3d

from soph.tasks.subtasks.subtask_base import BaseSubtask

from igibson.utils.mesh_util import quat2rotmat, xyzw2wxyz
from transforms3d.euler import euler2quat

from soph.utils.utils import bbox, pixel_to_point
from soph.utils.pcd_dict import PointCloudDict

class ScanSubtask(BaseSubtask):
    """
    Object Detection Task
    The goal is to detect an object with the robot camera
    """

    def __init__(self, env):
        super(ScanSubtask, self).__init__(env)

        self.initial_pos = np.array(self.config.get("initial_pos", [0, 0, 0]))
        self.initial_orn = np.array(self.config.get("initial_orn", [0, 0, 0]))


        self.goal_id = self.config.get("goal_id", 323)
        self.rot_dir = self.config.get("rot_dir", "clockwise")

    def reset_variables(self, env):
        self.point_cloud = PointCloudDict(precision=1, sub_precision=1)
        return


    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        return self.point_cloud.point_array()

    def step(self, env):
        """
        Perform task-specific step: step visualization and aggregate path length

        :param env: environment instance
        """

        state = env.get_state()
        seg = state["seg"]
        detections = np.unique(seg)
        if self.goal_id in detections:
            rmin, rmax, cmin, cmax = bbox(seg == self.goal_id)      
            depth = state["depth"]
        
            for r in range(rmin, rmax+1):
                for c in range(cmin, cmax + 1):
                    if seg[r,c,0] != self.goal_id: continue
                    d = depth[r,c,0]
                    if d == 0: continue
                   
                    point_in_wf = pixel_to_point(env, r, c, d)
                    self.point_cloud.insert(point_in_wf[:3])



