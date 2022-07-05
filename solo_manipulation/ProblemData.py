import numpy as np
import example_robot_data as erd
import pinocchio as pin

class problemDataAbstract:
    def __init__(self, frozen_names = []):
        self.dt = 0.015 # OCP dt
        self.dt_sim = 0.001
        self.dt_bldc = 0.0005
        self.r1 = int(self.dt / self.dt_sim)
        self.r2 = int(self.dt_sim / self.dt_bldc)
        self.init_steps = 10 # full stand phase
        self.target_steps = 30 # manipulation steps
        self.T = self.init_steps + self.target_steps -1

        self.robot = erd.load("solo12")
        self.q0 = self.robot.q0

        self.model = self.robot.model
        self.rdata = self.model.createData()
        self.collision_model = self.robot.collision_model
        self.visual_model = self.robot.visual_model

        self.frozen_names = frozen_names
        if frozen_names:
            self.frozen_idxs = [self.model.getJointId(id) for id in frozen_names]
            self.freeze()

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nx = self.nq + self.nv
        self.ndx = 2*self.nv
        self.nu = 12 - len(frozen_names)  + 1 if len(frozen_names) != 0 else 12 # -1 to take into account the freeflyer
        self.ntau = self.nv

        self.effort_limit = np.ones(self.nu) *3   

        self.v0 = np.zeros(self.nv)
        self.x0 = np.concatenate([self.q0, self.v0])
        self.u0 = np.zeros(self.nu)

        self.xref = self.x0
        self.uref = self.u0
                 
        self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot = 'FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT'
        self.cnames = [self.lfFoot, self.rfFoot, self.lhFoot, self.rhFoot]
        self.allContactIds = [ self.model.getFrameId(f) for f in self.cnames]
        self.lfFootId = self.model.getFrameId(self.lfFoot)
        self.rfFootId = self.model.getFrameId(self.rfFoot)
        self.lhFootId = self.model.getFrameId(self.lhFoot)
        self.rhFootId = self.model.getFrameId(self.rhFoot)

        self.Rsurf = np.eye(3)

    def freeze(self):
        geom_models = [self.visual_model, self.collision_model]
        self.model, geometric_models_reduced = pin.buildReducedModel(
                                                self.model,
                                                list_of_geom_models=geom_models,
                                                list_of_joints_to_lock=self.frozen_idxs,
                                                reference_configuration=self.q0) 
        self.rdata = self.model.createData()
        self.visual_model = geometric_models_reduced[0]
        self.collision_model = geometric_models_reduced[1]

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
        frozen_names = ["root_joint", "FL_HAA", "FL_HFE", "FL_KFE",
                        "HL_HAA", "HL_HFE", "HL_KFE",
                        "HR_HAA", "HR_HFE", "HR_KFE" ]

        super().__init__(frozen_names)
        
        self.useFixedBase = 1

        # Cost function weights
        self.mu = 0.7
        self.foot_tracking_w = 1e2
        #self.friction_cone_w = 1e3 * 0
        self.control_bound_w = 1e3
        self.control_reg_w = 1e1
        self.state_reg_w = np.array([1e-3] * 3 + [1e-1]*3)
        self.terminal_velocity_w = np.array([0] * 3 + [1e3] * 3 )

        self.q0 = self.q0[10:13]
        self.x0 = np.concatenate([self.q0, self.v0])

        self.xref = self.x0
        self.uref = self.u0
    
    

    