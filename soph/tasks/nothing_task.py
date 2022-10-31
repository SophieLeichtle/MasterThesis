import numpy as np

from igibson.tasks.task_base import BaseTask
from igibson.termination_conditions.max_collision import MaxCollision
from igibson.termination_conditions.out_of_bound import OutOfBound
from igibson.termination_conditions.timeout import Timeout
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene

class NothingTask(BaseTask):
    """
    Task that does nothing
    """

    def __init__(self, env):
        super(NothingTask, self).__init__(env)
        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
        ]

        self.initial_pos = np.array(self.config.get("initial_pos", [0, 0, 0]))
        self.initial_orn = np.array(self.config.get("initial_orn", [0, 0, 0]))


    def reset_scene(self, env):
        """
        Task-specific scene reset: reset scene objects or floor plane

        :param env: environment instance
        """
        if isinstance(env.scene, InteractiveIndoorScene):
            env.scene.reset_scene_objects()
        elif isinstance(env.scene, StaticIndoorScene):
            env.scene.reset_floor(floor=self.floor_num)

    def reset_agent(self, env):
        """
        Task-specific agent reset: land the robot to initial pose, compute initial potential

        :param env: environment instance
        """
        env.land(env.robots[0], self.initial_pos, self.initial_orn)

    def reset_variables(self, env):
        return

    def get_termination(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate termination conditions and fill info
        """
        
        return False, {}

    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """
       
        return []

    def step(self, env):
        """
        Perform task-specific step: step visualization and aggregate path length

        :param env: environment instance
        """
        return