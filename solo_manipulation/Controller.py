from shutil import which

from zmq import device
from ProblemData import ProblemData
from Target import Target
from utils.PyBulletSimulator import PyBulletSimulator
import numpy as np
from sklearn.linear_model import LinearRegression

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

    def compute_step(self, x0, loadPreviousSol = False, guess=None):
            if guess:
                self.ocp.solve(x0, guess=guess)
            elif loadPreviousSol:
                self.ocp.solve(x0, guess=self.last_result)
            else:
                self.ocp.solve(x0)

            _, x, a, u, f, _ = self.ocp.get_results()
            

            #print("Difference between starting point and initial state: ", x0 - x[0])

            #t = np.array([self.pd.dt*i for i in range(len(x)-1)])

            #self.last_result['xs'] = x[1:]  + [ (x[-1]- x[-2])/2]        
            #self.last_result['acs'] = a[1:] + [ (a[-1]- a[-2])/2]        
            #self.last_result['us'] = u[1:]  + [ (u[-1]- u[-2])/2]        
            #self.last_result['fs'] = f[1:]  + [ (f[-1]- f[-2])/2] 

            # With this it's working
            
            self.last_result['xs'] = x[1:]  + [x[-1] * 0]            
            self.last_result['acs'] = a[1:] + [a[-1] * 0]                      
            self.last_result['us'] = u[1:]  + [u[-1] * 0]
            if self.pd.useFixedBase == 0:           
                self.last_result['fs'] = f[1:]  + [f[-1] * 0]
            
            # Use this to break 
            #self.last_result['xs'] = x[1:]  + [x[-1]]           
            #self.last_result['acs'] = a[1:] + [a[-1]]           
            #self.last_result['us'] = u[1:]  + [u[-1]]           
            #self.last_result['fs'] = f[1:]  + [f[-1]]      
            
            #self.last_result['xs'] = x[1:]  + [x[-1] + (x[-1]- x[-2])/2]           
            #self.last_result['acs'] = a[1:] + [a[-1] + (a[-1]- a[-2])/2]           
            #self.last_result['us'] = u[1:]  + [u[-1] + (u[-1]- u[-2])/2]           
            #self.last_result['fs'] = f[1:]  + [f[-1] + (f[-1]- f[-2])/2]         

            self.results.x = np.array(x[0])
            self.results.u = np.array(u[0])
            if self.solver == 'crocoddyl':
                self.results.k = self.ocp.ddp.K[0]

            self.results.ocp_storage['fs']  += [f]
            self.results.ocp_storage['xs']  += [np.array(x)]
            self.results.ocp_storage['acs'] += [np.array(a)]
            self.results.ocp_storage['us']  += [np.array(u)]

    
                
    def get_q_mpc(self):
        q_mpc = []
        [q_mpc.append(x[1][: self.pd.nq]) for x in self.results.ocp_storage['xs']]
        return np.array(q_mpc)

    def get_q_sim_mpc(self):
        return self.results.x_m[:, : self.pd.nq]
