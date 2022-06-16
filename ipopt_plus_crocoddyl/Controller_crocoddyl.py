from zmq import XPUB_WELCOME_MSG
import crocoddyl
from utils.loader import Solo12
import numpy as np
import utils.ocp_manipulation_crocoddyl as ocp

class Results:
    def __init__(self):
        # Optimal state and control to be followed by LLC
        self.qj_des = []
        self.vj_des =  []
        self.tau_ff =  []
        self.tau = []
        self.x_m = []
        self.ocp_storage = {'xs': [], 'us': [], 'fw': [], 'qj_des': [], 'vj_des': [], 'residuals' : {'inf_pr': [], 'inf_du': []}}


class Controller:
    def __init__(self, init_steps, target_steps, dt):
        self.dt = dt
        self.init_steps = init_steps
        self.target_steps = target_steps
        self.n_nodes = self.init_steps + self.target_steps

        self.solo = Solo12()
        self.results = Results()

        self.nq = self.solo.nq 
        nv = self.solo.nv
        self.q0 = self.solo.q0
        self.qj0 = self.q0[7:self.nq]
        v0 = np.zeros(nv)
        self.x0 = np.array([ 0.        ,  0.        ,  0.23289725,  0.        ,  0.        ,
        0.        ,  1.        ,  0.1       ,  0.8       , -1.6       ,
       -0.1       ,  0.8       , -1.6       ,  0.1       , -0.8       ,
        1.6       , -0.1       , -0.8       ,  1.6       ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ])
        self.u0 = np.array([-0.02615051, -0.25848605,  0.51696646,  0.0285894 , -0.25720605,
                            0.51441775, -0.02614404,  0.25848271, -0.51697107,  0.02859587,
                            0.25720939, -0.51441314])

        self.results.ocp_storage['xs'] += [np.array([self.x0] * self.n_nodes)]

        self.warmstart = {'xs': [], 'acs': [], 'us':[], 'fs': []}

        self.gait = [] \
            + [ [ 1,1,1,1 ] ] * init_steps \
            + [ [ 1,0,1,1 ] ] * target_steps

        lfFoot, rfFoot, lhFoot, rhFoot = 'FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT'
        self.ocp = ocp.SimpleManipulationProblem(self.solo.model, lfFoot, rfFoot, lhFoot, rhFoot )

    def create_target(self, t_):
        # Here we create the target of the control
        # FR Foot moving in a circular way

        FR_foot0 = np.array([0.1946, -0.16891, 0.017])
        A = np.array([0, 0.035, 0.035]) 
        offset = np.array([0.05, 0., 0.06])
        freq = np.array([0, 2.5, 2.5])
        phase = np.array([0,0,np.pi/2])

        target = []
        for t in range(self.n_nodes): target += [FR_foot0 + offset +A*np.sin(2*np.pi*freq * (t+t_)* self.dt + phase)]
        self.target = np.array(target)

    def shift_gate(self):
        self.gait.pop(0)
        self.gait += [self.gait[-1]]

    def compute_step(self, x0, x_ref, u_ref, guess=None):
        problem = self.ocp.createMovingFootProblem(self.x0, x_ref, u_ref, self.gait, self.target, self.dt)

        self._solver = crocoddyl.SolverDDP(problem)

        # Solve the DDP problem
        print('*** SOLVE ***')
        self._solver.setCallbacks([crocoddyl.CallbackVerbose()])

        if guess:
            xs = guess['xs']
            us = guess['us']
            
        else:
            xs = [x0] * (self._solver.problem.T + 1)
            us = self._solver.problem.quasiStatic([x0] * self._solver.problem.T)
        self._solver.solve(xs, us, 100, False, 0.1)

        self.results.ocp_storage['xs'] += [np.array(self._solver.xs.tolist())]
        self.results.ocp_storage['fw'] += [self.get_croco_forces()]
        self.results.ocp_storage['us'] += [np.array(self._solver.us.tolist())]
        self.results.ocp_storage['xs'] += [np.array(self._solver.xs.tolist())]


        """ self.ocp.solve(guess=self.warmstart)
        _, x, a, u, f, fw = self.ocp.get_results()  
        self.warmstart['xs'] = x[1:]
        self.warmstart['acs'] = a[1:]
        self.warmstart['us'] = u[1:]
        self.warmstart['fs'] = f[1:]

        self.results.ocp_storage['fw'] += [fw]
        self.results.ocp_storage['xs'] += [x]
        self.results.ocp_storage['us'] += [u]
        self.results.ocp_storage['qj_des'] += [x[:, 7: self.nq]]
        self.results.ocp_storage['vj_des'] += [x[:, self.nq + 6: ]]

        self.results.qj_des += [x[:, 7: self.nq][1]]
        self.results.vj_des += [x[:, self.nq + 6: ][1]]
        self.results.tau_ff += [u[0]]

        self.results.ocp_storage['residuals']['inf_pr'] += [self.ocp.opti.stats()['iterations']['inf_pr']]
        self.results.ocp_storage['residuals']['inf_du'] += [self.ocp.opti.stats()['iterations']['inf_du']] """

    def get_croco_forces(self):
        d = self._solver.problem.runningDatas[0]
        cnames = d.differential.multibody.contacts.contacts.todict().keys()
        forces = {n : [] for n in cnames}

        for m in self._solver.problem.runningDatas:
            mdict = m.differential.multibody.contacts.contacts.todict()
            for n in cnames:
                if n in mdict:
                    forces[n] += [(mdict[n].jMf.inverse()*mdict[n].f).linear]
                else:
                    forces[n] += [np.array([0,0,0])]
        for f in forces: forces[f] = np.array(forces[f])
        return forces

    def get_croco_forces_ws(self):
        forces = []

        for m in self._solver.problem.runningDatas:
            mdict = m.differential.multibody.contacts.contacts.todict()
            f_tmp = []
            for n in mdict:
                f_tmp += [(mdict[n].jMf.inverse()*mdict[n].f).linear]
            forces += [np.concatenate(f_tmp)]
        return forces

    def get_croco_acc(self):
        acc = []
        [acc.append(m.differential.xout) for m in self._solver.problem.runningDatas ]
        return acc





    


