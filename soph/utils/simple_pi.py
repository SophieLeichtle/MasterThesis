import numpy as np

class SimplePI:

    def __init__(self, Kp, Ki, timestep):
        self.Kp = Kp
        self.Ki = Ki
        self.timestep = timestep
        self.precision = 0.001
    
    def reset_goal(self, goal, state):
        self.goal = goal
        self.cumError = 0
        error = self.goal - state
        if abs(error[2]) > self.precision: 
            self.angular = True
        else:
            self.angular = False
    
    def get_control(self, state):
        error = self.goal - state
        self.cumError += np.linalg.norm(error) * self.timestep
        if not self.angular:
            lin_vel = self.Kp * np.linalg.norm(error[:2]) - self.Ki * self.cumError
            if lin_vel > 1: return [1., 0]
            if lin_vel < -1: return [-1., 0]
            return [lin_vel, 0]
        else:
            ang_vel = - self.Kp * error[2] - self.Ki * self.cumError
            if ang_vel > 1: return [0, 1.]
            if ang_vel < -1: return [0, -1.]
            return [0,ang_vel]
