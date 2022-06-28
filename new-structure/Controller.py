from shutil import which
from utils.PyBulletSimulator import PyBulletSimulator
import numpy as np

class OCPData:
    def __init__(self):
        self.ocp_storage = {'xs': [], 'acs': [], 'us': [], 'fs': [], 'qj_des': [], 'vj_des': [], 'residuals' : {'inf_pr': [], 'inf_du': []}}

class Controller:
    def __init__(self, problemData, dt_sim, r, solver = 'ipopt'):
        self.pd = problemData
        self.results = OCPData()
        if solver == 'ipopt':
            from CasadiOCP import CasadiOCP as OCP
        elif solver == 'crocoddyl':
            from CrocoddylOCP import CrocoddylOCP as OCP

        self.ocp = OCP(self.pd)
        self.warmstart = {'xs': [], 'acs': [], 'us':[], 'fs': []}

        self.solver = solver
        self.dt_sim = dt_sim
        self.r = r
        self.device = self.Init_simulation(self.pd.q0[7:])

    def Init_simulation(self, q_init):
        device = PyBulletSimulator()
        device.Init(q_init, 0, True, True, self.dt_sim)
        return device

    def tuple_to_array(self, tup):
        a = np.array([element for tupl in tup for element in tupl])
        return a

    def interpolate_traj(self, q_des, v_des):
        measures = self.read_state()
        qj_des_i = np.linspace(measures['qj_m'], q_des, self.r)
        vj_des_i = np.linspace(measures['vj_m'], v_des, self.r)

        return qj_des_i, vj_des_i

    def read_state(self):
        self.device.parse_sensor_data()
        qj_m = self.device.joints.positions
        vj_m = self.device.joints.velocities
        bp_m = self.tuple_to_array(self.device.baseState)
        bv_m = self.tuple_to_array(self.device.baseVel)
        x_m = np.concatenate([bp_m, qj_m, bv_m, vj_m])
        return {'qj_m': qj_m, 'vj_m': vj_m, 'x_m': x_m}

    def store_measures(self, data_storage, all=True):
        m = self.read_state()
        data_storage.x_m += [m['x_m']]
        if all == True:
            data_storage.tau += [self.device.jointTorques]

    def send_torques(self, x, u, k):
        if self.solver == 'crocoddyl':
            for t in range(self.r):
                m = self.read_state()
                feedback = np.dot(k, solver.state.diff(m['x_m'], x))
                self.device.joints.set_torques(u + feedback)
                self.device.send_command_and_wait_end_of_cycle()

            self.store_measures()

    def compute_step(self, x0, guess={}):
        if guess:
            self.warmstart['xs'] = guess['xs']
            self.warmstart['us'] = guess['us']
            if self.solver == 'ipopt':
                self.warmstart['acs'] = guess['acs']
                self.warmstart['fs'] = guess['fs']

        self.ocp.solve(x0, guess=self.warmstart)
        _, x, a, u, f, fw = self.ocp.get_results()

        self.warmstart['xs'] = x[1:]  + [x[-1]]
        self.warmstart['acs'] = a[1:] + [a[-1]]
        self.warmstart['us'] = u[1:]  + [u[-1]]
        self.warmstart['f_ws'] = f[1:]  + [f[-1]]

        self.results.ocp_storage['fs']  += [f]
        self.results.ocp_storage['xs']  += [np.array(x)]
        self.results.ocp_storage['acs'] += [np.array(a)]
        self.results.ocp_storage['us']  += [np.array(u)]
