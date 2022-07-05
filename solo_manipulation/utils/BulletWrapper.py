from .PyBulletSimulator import PyBulletSimulator
import numpy as np

class BulletWrapper:
    def __init__(self, ctrl):
            self.device = PyBulletSimulator()
            self.ctrl = ctrl
            q0 = ctrl.pd.q0[7: ]

            self.device.Init(q0, 0, True, True,  ctrl.pd.useFixedBase, ctrl.pd.dt_sim)

    def read_state(self):
        self.device.parse_sensor_data()
        qj_m = self.device.joints.positions
        vj_m = self.device.joints.velocities
        bp_m = self.tuple_to_array(self.device.baseState)
        bv_m = self.tuple_to_array(self.device.baseVel)
        if self.ctrl.pd.useFixedBase == 0:
            x_m = np.concatenate([bp_m, qj_m, bv_m, vj_m])
        else:
            x_m = np.concatenate([qj_m[3:6], vj_m[3:6]])

        return {'qj_m': qj_m, 'vj_m': vj_m, 'x_m': x_m}

    def store_measures(self, all=True):
        m = self.read_state()
        self.ctrl.results.x_m += [m['x_m']]
        if all == True:
            self.ctrl.results.u_m += [self.device.jointTorques]

    def send_torques(self, x, u, k=None):
        if self.ctrl.solver == 'crocoddyl':
            q = x[:3]
            v = x[3:]
            q_des, v_des = self.interpolate_traj(q, v, self.ctrl.pd.r1)
            for t in range(self.ctrl.pd.r1):
                m = self.read_state()
                q_ref = self.ctrl.pd.q0[7:]
                q_ref[3:6] = q_des[t]
                v_ref = self.ctrl.pd.v0[6:]
                v_ref[3:6] = v_des[t]
                self.device.joints.set_desired_positions(q_ref)
                self.device.joints.set_desired_velocities(v_ref)
                self.device.joints.set_position_gains(100)
                self.device.joints.set_velocity_gains(0.01)
                self.device.send_command_and_wait_end_of_cycle()
                self.store_measures()

            #for t in range(self.ctrl.pd.r1):
            #    m = self.read_state()
            #    feedback = np.dot(k, self.ctrl.ocp.state.diff(m['x_m'], x))
            #    tau_actuated = u + feedback
            #    tau0 = np.zeros(3)
            #    self.device.joints.set_torques(np.concatenate([tau0, tau_actuated, tau0, tau0]))
            #    self.device.send_command_and_wait_end_of_cycle()
            #    self.store_measures()

        if self.ctrl.solver == 'ipopt':
            q, v = self.x2qv(x)
            q_des, v_des = self.interpolate_traj(q, v, self.ctrl.pd.r1)
            for t in range(self.ctrl.pd.r1):
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
        q = x[7: self.ctrl.pd.nq]
        v = x[self.ctrl.pd.nq + 6 :]
        return q, v

    def interpolate_traj(self, q_des, v_des, ratio):
        measures = self.read_state()
        qj_des_i = np.linspace(measures['qj_m'][3:6], q_des, ratio)
        vj_des_i = np.linspace(measures['vj_m'][3:6], v_des, ratio)

        return qj_des_i, vj_des_i