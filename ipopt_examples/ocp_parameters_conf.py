import numpy as np
import example_robot_data as robex
# OCP parameters
mu = 0.7

foot_tracking_w = 1e5
force_reg_w = 1e-2
control_reg_w = 1e1
state_reg_w = np.array([0] * 3 \
                    + [1e1] * 3 \
                    + [1e0] * 3 \
                    + [1e-2] * 3\
                    + [1e0] * 6
                    + [0] * 6 \
                    + [1e1] * 3 \
                    + [1e0] * 3\
                    + [1e1] * 6 )  
friction_cone_w = 1e8
terminal_velocity_w = 1e5


 
