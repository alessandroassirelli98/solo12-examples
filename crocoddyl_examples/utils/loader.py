import example_robot_data as erd
import pinocchio as pin
import numpy as np

class Solo12:
    def __init__(self):
        self.robot = erd.load("solo12")
        self.model = self.robot.model
        self.q0 = self.robot.q0
        self.nq = self.robot.nq
        self.nv = self.robot.nv
        self.v0 = np.zeros(self.nv)