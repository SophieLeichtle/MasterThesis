from abc import ABCMeta, abstractmethod


class BaseSubtask:
    """
    Base Subtask class.
    Task-specific get_task_obs, step methods are implemented in subclasses
    """

    __metaclass__ = ABCMeta

    def __init__(self, env):
        self.config = env.config

    def reset_variables(self, env):
        """
        Task-specific variable reset

        :param env: environment instance
        """
        return

    def reset(self, env):
        self.reset_variables(env)


    @abstractmethod
    def get_task_obs(self, env):
        """
        Get task-specific observation

        :param env: environment instance
        :return: task-specific observation (numpy array)
        """
        raise NotImplementedError()

    def step(self, env):
        """
        Perform task-specific step for every timestep

        :param env: environment instance
        """
        return
