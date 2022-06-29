import numpy as np
import example_robot_data as erd
import pinocchio as pin

class ProblemData:
    def __init__(self):
        self.dt = 0.015 # OCP dt
        self.init_steps = 10 # full stand phase
        self.target_steps = 50 # manipulation steps
        self.T = self.init_steps + self.target_steps -1

        # Cost function weights
        self.mu = 0.7
        self.foot_tracking_w = 1e5
        self.control_reg_w = 1e1
        self.state_reg_w = np.array([0] * 3 \
                            + [1e1] * 3 \
                            + [1e0] * 3 \
                            + [1e-2] * 3\
                            + [1e0] * 6
                            + [0] * 6 \
                            + [1e1] * 3 \
                            + [1e0] * 3\
                            + [1e1] * 6 ) 
        self.terminal_velocity_w = np.array([0] * 18 + [1e3] * 18 )
        self.friction_cone_w = 1e3

        self.robot = erd.load("solo12")
        self.model = self.robot.model
        self.rdata = self.model.createData()
        self.q0 = self.robot.q0
        self.nq = self.robot.nq
        self.nv = self.robot.nv
        self.nx = self.nq + self.nv
        self.ndx = 2*self.nv
        self.nu = self.nv-6
        self.ntau = self.nv

        self.effort_limit = np.ones(self.nu) *3   

        self.v0 = np.zeros(self.nv)
        self.x0 = np.array([ 0, 0, 0.23289725, 0, 0, 0, 1, 0.1, 0.8, -1.6,
                            -0.1,  0.8, -1.6,  0.1, -0.8, 1.6, -0.1, -0.8, 1.6,
                            0, 0, 0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # x0 got from PyBullet
        self.u0 = np.array([-0.02615051, -0.25848605,  0.51696646,  
                            0.0285894 , -0.25720605, 0.51441775, 
                            -0.02614404, 0.25848271, -0.51697107,  
                            0.02859587, 0.25720939, -0.51441314]) # quasi static control
        self.xref = self.x0
        self.uref = self.u0

        self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot = 'FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT'
        cnames = [self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot]
        self.allContactIds = [ self.model.getFrameId(f) for f in cnames]
        self.lfFootId = self.model.getFrameId(self.lfFoot)
        self.rfFootId = self.model.getFrameId(self.rfFoot)
        self.lhFootId = self.model.getFrameId(self.lhFoot)
        self.rhFootId = self.model.getFrameId(self.rhFoot)

        self.Rsurf = np.eye(3)

class Target:
    def __init__(self, pd:ProblemData):
        self.pd = pd
        self.dt = pd.dt

        self.gait = [] \
            + [ [ 1,1,1,1 ] ] * pd.init_steps \
            + [ [ 1,0,1,1 ] ] * pd.target_steps

        self.T = pd.T
        self.contactSequence = [ self.patternToId(p) for p in self.gait]

        self.target = {pd.rfFootId: []}
        self.FR_foot0 = np.array([0.1946, -0.16891, 0.017])
        self.A = np.array([0, 0.035, 0.035])
        self.offset = np.array([0.05, 0., 0.06])
        self.freq = np.array([0, 2.5, 2.5])
        self.phase = np.array([0, 0, np.pi/2])

    def patternToId(self, gait):
            return tuple(self.pd.allContactIds[i] for i,c in enumerate(gait) if c==1 )
        
    def shift_gait(self):
        self.gait.pop(0)
        self.gait += [self.gait[-1]]
        self.contactSequence = [ self.patternToId(p) for p in self.gait]    

    def update(self, t):
        target = []
        for n in range(self.T+1):
            target += [self.FR_foot0 + self.offset + self.A *
                       np.sin(2*np.pi*self.freq * (n+t) * self.dt + self.phase)]
        self.target[self.pd.rfFootId] = np.array(target)
    
    def evaluate_in_t(self, t):
        return {e: self.target[e][t] for e in self.target}

    