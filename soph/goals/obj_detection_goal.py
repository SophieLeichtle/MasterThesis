from igibson.termination_conditions.termination_condition_base import BaseTerminationCondition
import numpy as np

class ObjectDetectionGoal(BaseTerminationCondition):
    """
    Goal to detect object with robot camera
    """

    def __init__(self, config):
        super(ObjectDetectionGoal, self).__init__(config)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate
        Terminate if object detected
        """
        state = env.get_state()

        seg = state["seg"]
        detections = np.unique(seg)
        count = 0.0
        if task.goal_id in detections:
            count = np.count_nonzero(seg == task.goal_id)
        done = count >= 1000
        success = done
        return done, success