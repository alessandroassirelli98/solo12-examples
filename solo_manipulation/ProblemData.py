import numpy as np
import example_robot_data as erd
import pinocchio as pin

class problemDataAbstract:
    def __init__(self):
        self.dt = 0.015 # OCP dt
        self.dt_sim = 0.001
        self.dt_bldc = 0.0005
        self.r1 = int(self.dt / self.dt_sim)
        self.r2 = int(self.dt_sim / self.dt_bldc)
        self.init_steps = 10 # full stand phase
        self.target_steps = 20 # manipulation steps
        self.T = self.init_steps + self.target_steps -1

        self.robot = erd.load("solo12")
        self.model = self.robot.model
        self.rdata = self.model.createData()
        self.collision_model = self.robot.collision_model
        self.visual_model = self.robot.visual_model
        self.q0 = self.robot.q0
        self.nq = self.robot.nq
        self.nv = self.robot.nv
        self.nx = self.nq + self.nv
        self.ndx = 2*self.nv
        self.nu = 12
        self.ntau = 12

        self.effort_limit = np.ones(12) *3   

        self.v0 = np.zeros(self.nv)
                            

        self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot = 'FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT'
        self.cnames = [self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot]
        self.allContactIds = [ self.model.getFrameId(f) for f in self.cnames]
        self.lfFootId = self.model.getFrameId(self.lfFoot)
        self.rfFootId = self.model.getFrameId(self.rfFoot)
        self.lhFootId = self.model.getFrameId(self.lhFoot)
        self.rhFootId = self.model.getFrameId(self.rhFoot)

        self.Rsurf = np.eye(3)

class ProblemData(problemDataAbstract):
    def __init__(self):
        super().__init__()
        
        self.useFixedBase = 0
        # Cost function weights
        self.mu = 0.7
        self.foot_tracking_w = 1e2
        self.friction_cone_w = 1e3
        self.control_bound_w = 1e3
        self.control_reg_w = 1e1
        self.state_reg_w = np.array([0] * 3 \
                            + [1e1] * 3 \
                            + [1e0] * 3 \
                            + [1e-3] * 3\
                            + [1e0] * 6
                            + [0] * 6 \
                            + [1e1] * 3 \
                            + [1e-1] * 3\
                            + [1e1] * 6 ) 
        self.terminal_velocity_w = np.array([0] * 18 + [1e3] * 18 )
        self.control_bound_w = 1e3

        self.x0 = np.array([ 0, 0, 0.23289725, 0, 0, 0, 1, 0.1, 0.8, -1.6,
                            -0.1,  0.8, -1.6,  0.1, -0.8, 1.6, -0.1, -0.8, 1.6,
                            0, 0, 0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # x0 got from PyBullet
                            
        self.u0 = np.array([-0.02615051, -0.25848605,  0.51696646,  
                            0.0285894 , -0.25720605, 0.51441775, 
                            -0.02614404, 0.25848271, -0.51697107,  
                            0.02859587, 0.25720939, -0.51441314]) # quasi static control
        self.xref = self.x0
        self.uref = self.u0


class ProblemDataFull(problemDataAbstract):
    def __init__(self):
        super().__init__()
        
        self.useFixedBase = 1

        # Cost function weights
        self.mu = 0.7
        self.foot_tracking_w = 1e2 
        self.friction_cone_w = 1e3

        self.control_bound_w = 1e3
        self.control_reg_w = 1e1
        self.state_reg_w = np.array( [1e0] * 3 \
                            + [1e0] * 3\
                            + [1e0] * 6
                            + [1e1] * 3 \
                            + [1e-1] * 3\
                            + [1e1] * 6 ) 
        self.terminal_velocity_w = np.array([0] * 12 + [1e3] * 12 ) *0

        self.x0 = np.array([ 0.1, 0.8, -1.6,
                            -0.1,  0.8, -1.6,  0.1, -0.8, 1.6, -0.1, -0.8, 1.6,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # x0 got from PyBullet
                            
        self.u0 = np.zeros(self.nu)

        self.xref = self.x0
        self.uref = self.u0

        frozen_names = ["FL_HAA", "FL_HFE", "FL_KFE",
                            "HL_HAA", "HL_HFE", "HL_KFE",
                            "HR_HAA", "HR_HFE", "HR_KFE" ]
        self.frozen_idxs = [self.model.getJointId(id) for id in frozen_names]
        

    
    
    def freeze(self, idxs):
        pass

    