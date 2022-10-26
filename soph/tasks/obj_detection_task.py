from lib2to3.pytree import Base
import numpy as np

from igibson.tasks.task_base import BaseTask
from igibson.termination_conditions.max_collision import MaxCollision
from igibson.termination_conditions.out_of_bound import OutOfBound
from igibson.termination_conditions.timeout import Timeout
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene

from igibson.reward_functions.collision_reward import CollisionReward

from igibson.utils.utils import rotate_vector_3d, l2_distance


from soph.goals.obj_detection_goal import ObjectDetectionGoal
from soph.rewards.obj_detection_reward import ObjectDetectionReward

class ObjectDetectionTask(BaseTask):
    """
    Object Detection Task
    The goal is to detect an object with the robot camera
    """

    def __init__(self, env):
        super(ObjectDetectionTask, self).__init__(env)
        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
            ObjectDetectionGoal(self.config),
        ]
        self.reward_functions = [
            CollisionReward(self.config),
            #ObjectDetectionReward(self.config),
        ]

        self.initial_pos = np.array(self.config.get("initial_pos", [0, 0, 0]))
        self.initial_orn = np.array(self.config.get("initial_orn", [0, 0, 0]))

        self.goal_id = self.config.get("goal_id", 323)


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
        self.path_length = 0.0
        self.robot_pos = self.initial_pos[:2]

    def get_termination(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate termination conditions and fill info
        """
        done, info = super(ObjectDetectionTask, self).get_termination(env, collision_links, action, info)

        info["path_length"] = self.path_length

        return done, info

    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        # linear velocity along the x-axis
        linear_velocity = rotate_vector_3d(env.robots[0].get_linear_velocity(), *env.robots[0].get_rpy())[0]
        # angular velocity along the z-axis
        angular_velocity = rotate_vector_3d(env.robots[0].get_angular_velocity(), *env.robots[0].get_rpy())[2]
        task_obs = [linear_velocity, angular_velocity]

        return task_obs

    def step(self, env):
        """
        Perform task-specific step: step visualization and aggregate path length

        :param env: environment instance
        """
        new_robot_pos = env.robots[0].get_position()[:2]
        self.path_length += l2_distance(self.robot_pos, new_robot_pos)
        self.robot_pos = new_robot_pos