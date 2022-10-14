from igibson.reward_functions.reward_function_base import BaseRewardFunction
import numpy as np

class ObjectDetectionReward(BaseRewardFunction):
    """
    Object Detection Reward
    """

    def __init__(self, config):
        super(ObjectDetectionReward, self).__init__(config)
        self.success_reward = self.config.get("success_reward", 10.0)

    def get_reward(self, task, env):
        """
        Check if object detected by robot camera
        """

        state = env.get_state()

        seg = state["seg"]
        detections = np.unique(seg)
        count = 0.0
        if task.goal_id in detections:
            count = np.count_nonzero(seg == task.goal_id)
        reward = count / 1000.0 * self.success_reward
        return reward