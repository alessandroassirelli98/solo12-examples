from shutil import which

from zmq import device
from ProblemData import ProblemData, Target
from utils.PyBulletSimulator import PyBulletSimulator
import numpy as np

class SimulationData:
    def __init__(self):
        # Optimization data
        self.x = None
        self.u = None
        self.k = None
        self.ocp_storage = {'xs': [], 'acs': [], 'us': [], 'fs': [], 'qj_des': [], 'vj_des': [], 'residuals' : {'inf_pr': [], 'inf_du': []}}

        # Measured data
        self.x_m = []
        self.u_m = []

    def make_arrays(self):   
        self.x_m = np.array(self.x_m)
        self.u_m = np.array(self.u_m)

class Controller:
    def __init__(self, pd:ProblemData, target:Target, solver):
        self.pd = pd

        self.results = SimulationData()
        if solver == 'ipopt':
            from CasadiOCP import CasadiOCP as OCP
        elif solver == 'crocoddyl':
            from CrocoddylOCP import CrocoddylOCP as OCP

        self.ocp = OCP(pd, target)
        self.last_result = {'xs': [], 'acs': [], 'us':[], 'fs': []}

        self.solver = solver

        self.device = self.Init_simulation(pd.q0[7:])

    def compute_step(self, x0, guess=None):
            if guess:
                self.ocp.solve(x0, guess=guess)
            else:
                self.ocp.solve(x0, guess=self.last_result)

            _, x, a, u, f, _ = self.ocp.get_results()

            self.last_result['xs'] = x[1:]  + [x[-1]]
            self.last_result['acs'] = a[1:] + [a[-1]]
            self.last_result['us'] = u[1:]  + [u[-1]]
            self.last_result['fs'] = f[1:]  + [f[-1]]

            self.results.x = np.array(x[1])
            self.results.u = np.array(u[0])
            if self.solver == 'crocoddyl':
                self.results.k = self.ocp.ddp.K[0]

            self.results.ocp_storage['fs']  += [f]
            self.results.ocp_storage['xs']  += [np.array(x)]
            self.results.ocp_storage['acs'] += [np.array(a)]
            self.results.ocp_storage['us']  += [np.array(u)]

    def Init_simulation(self, q_init):
        device = PyBulletSimulator()
        device.Init(q_init, 0, True, True, self.pd.dt_sim)
        return device

    def read_state(self):
        self.device.parse_sensor_data()
        qj_m = self.device.joints.positions
        vj_m = self.device.joints.velocities
        bp_m = self.tuple_to_array(self.device.baseState)
        bv_m = self.tuple_to_array(self.device.baseVel)
        x_m = np.concatenate([bp_m, qj_m, bv_m, vj_m])
        return {'qj_m': qj_m, 'vj_m': vj_m, 'x_m': x_m}

    def store_measures(self, all=True):
        m = self.read_state()
        self.results.x_m += [m['x_m']]
        if all == True:
            self.results.u_m += [self.device.jointTorques]

    def send_torques(self, x, u, k=None):
        if self.solver == 'crocoddyl':
            for t in range(self.pd.r1):
                m = self.read_state()
                feedback = np.dot(k, self.ocp.state.diff(m['x_m'], x))
                self.device.joints.set_torques(u + feedback)
                self.device.send_command_and_wait_end_of_cycle()

        if self.solver == 'ipopt':
            q, v = self.x2qv(x)
            q_des, v_des = self.interpolate_traj(q, v, self.pd.r1)
            for t in range(self.pd.r1):
                self.device.joints.set_desired_positions(q_des[t])
                self.device.joints.set_desired_velocities(v_des[t])
                self.device.joints.set_position_gains(3)
                self.device.joints.set_velocity_gains(0.1)
                self.device.joints.set_torques(u)
                self.device.send_command_and_wait_end_of_cycle()
                self.store_measures()
                
    def tuple_to_array(self, tup):
        a = np.array([element for tupl in tup for element in tupl])
        return a
    
    def x2qv(self, x):
        q = x[7: self.pd.nq]
        v = x[self.pd.nq + 6 :]
        return q, v

    def interpolate_traj(self, q_des, v_des, ratio):
        measures = self.read_state()
        qj_des_i = np.linspace(measures['qj_m'], q_des, ratio)
        vj_des_i = np.linspace(measures['vj_m'], v_des, ratio)

        return qj_des_i, vj_des_i
                
    def get_q_mpc(self):
        q_mpc = []
        [q_mpc.append(x[1][: self.pd.nq]) for x in self.results.ocp_storage['xs']]
        return np.array(q_mpc)

    def get_q_sim_mpc(self):
        return self.results.x_m[:, : self.pd.nq]
