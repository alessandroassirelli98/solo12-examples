import os
from time import time

import casadi
import example_robot_data as robex
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
from pinocchio.visualize import GepettoVisualizer
from shooting_node_ipopt import ShootingNode
from data import ModelData
from problem import ProblemData

plt.style.use('seaborn')
path = os.getcwd()

class SimpleManipulationProblem():
    def __init__(self, problemData=ProblemData):
        """Define an optimal ocntrol problem.
        :param robot: Pinocchio RobotWrapper instance
        :param gait: list of list containing contact pattern i.e. for two steps[ [1,1,1,1], [1,0,0,1] ]. \
            Its length determines the horizon of the ocp
        :param x0: starting configuration
        :param x_ref: reference configuration
        :param target: array of target positions
        :param dt: timestep integration
        """
        

        self.problemData = problemData
        self.robot = problemData.robot
        self.model = model = problemData.model
        self.rdata = model.createData()
        self.cmodel = cpin.Model(model)

        self.iterationTime = 0

        self.effort_limit = np.ones(problemData.nu) *3   

        self.casadiActionModels = { contacts: ShootingNode( problemData=problemData, \
                                                            allContactIds=problemData.allContactIds,\
                                                            contactIds=contacts)  \
                                    for contacts in set(problemData.contactSequence) } # build the different shooting nodes
        
    def warmstart(self, x0, guess=None):           
        for g in guess:
            if guess[g] == []:
                print("No warmstart provided")         
                return 0
        
        try:
            xs_g = guess['xs']
            us_g = guess['us']
            acs_g = guess['acs']
            fs_g = guess['fs']

            def xdiff(x1,x2):
                nq = self.model.nq
                return np.concatenate([
                    pin.difference(self.model,x1[:nq],x2[:nq]), x2[nq:]-x1[nq:] ])

            for x,xg in zip(self.dxs,xs_g): self.opti.set_initial(x, xdiff(x0,xg))
            for a,ag in zip(self.acs,acs_g): self.opti.set_initial(a, ag)
            for u,ug in zip(self.us,us_g): self.opti.set_initial(u,ug)
            for f, fg in zip(self.fs, fs_g):
                fgsplit = np.split(fg, len(f))
                fgc = []
                [fgc.append(f) for f in fgsplit]
                [self.opti.set_initial(f[i], fgc[i]) for i in range(len(f))]
            print("Got warm start")
        except:
            print("Can't load warm start")
           
    def get_results(self):
        dxs_sol = np.array([self.opti.value(dx) for dx in self.dxs])
        xs_sol = np.array([self.opti.value(x) for x in self.xs])
        us_sol = np.array([self.opti.value(u) for u in self.us])
        acs_sol = np.array([self.opti.value(a) for a in self.acs])
        fsol = {name: [] for name in self.problemData.allContactIds}
        fs_world = {name: [] for name in self.problemData.allContactIds}
        fsol_to_ws = []
        for t in range(self.problemData.T):
            for i, st_foot in enumerate(self.runningModels[t].contactIds):
                fsol[st_foot].append(self.opti.value(self.fs[t][i]))
            for i, sw_foot in enumerate(self.runningModels[t].freeIds):
                fsol[sw_foot].append(np.zeros(3))

            fsol_to_ws.append(np.concatenate([self.opti.value(
                self.fs[t][j]) for j in range(len(self.runningModels[t].contactIds))]))
            
            pin.framesForwardKinematics(self.model, self.rdata, xs_sol[t, : self.problemData.nq])
            [fs_world[foot].append(self.rdata.oMf[foot].rotation @ fsol[foot][t]) for foot in fs_world]

        for foot in fs_world:
            fs_world[foot] = np.array(fs_world[foot])

        return dxs_sol, xs_sol, acs_sol, us_sol, fsol_to_ws, fs_world

    def get_feet_position(self, xs_sol):
        """ Get the feet positions computed durnig one ocp
        
        :param xs_sol: Array of dimension [n_steps, n_states]

        """
        feet_log = {i:[] for i in self.problemData.allContactIds}
        for foot in feet_log:
            tmp = []
            for i in range(len(xs_sol)):
                tmp += [ self.terminalModel.feet[foot](xs_sol[i]).full()[:, 0] ]
            feet_log[foot] = np.array(tmp)
        
        return feet_log

    def get_feet_velocity(self, xs_sol):
        """ Get the feet velocities in LOCAL_WORLD_ALIGNED frame computed durnig one ocp
        
        :param xs_sol: Array of dimension [n_steps, n_states]
        
        """
        feet_log = {i:[] for i in self.problemData.allContactIds}
        for foot in feet_log:
            tmp = []
            for i in range(len(xs_sol)):
                tmp += [ self.terminalModel.vfeet[foot](xs_sol[i]).full()[:, 0] ]
            feet_log[foot] = np.array(tmp)
        
        return feet_log

    def get_base_log(self, xs_sol):
        """ Get the base positions computed durnig one ocp
        
        :param xs_sol: Array of dimension [n_steps, n_states]
        
        """
        base_pos_log = []
        [base_pos_log.append(self.terminalModel.baseTranslation(xs_sol[i]).full()[:,0]) for i in range(len(xs_sol))]
        base_pos_log = np.array(base_pos_log)
        return base_pos_log

    def make_ocp(self):
        totalcost = 0
        eq = []

        eq.append(self.dxs[0])
        for t in range(self.problemData.T):
            print(self.problemData.contactSequence[t])
            self.runningModels[t].init(self.datas[t], self.xs[t], self.acs[t], self.us[t], self.fs[t], False) # These change every time! Do not use runningModels[0].x to get the state!!!! use xs[0]

            if (self.problemData.contactSequence[t] != self.problemData.contactSequence[t-1] and t >= 1): # If it is landing
                print('Contact on ', str(self.runningModels[t].contactIds)) 
                

            xnext,rcost = self.runningModels[t].calc(x_ref=self.problemData.xref,\
                                                    u_ref=self.problemData.uref, \
                                                    target=self.problemData.target[t])

            # Constraints
            eq.append(self.runningModels[t].constraint_standing_feet_eq() )
            eq.append(self.runningModels[t].constraint_dynamics_eq() )
            eq.append( self.runningModels[t].difference(self.xs[t + 1],xnext) - np.zeros(2*self.runningModels[t].nv) )

            totalcost += rcost

        eq_constraints = casadi.vertcat(*eq)

        return totalcost, eq_constraints

    def use_ipopt_solver(self, x0, guess = None):

        opti = casadi.Opti()
        self.opti = opti
        # Optimization variables
        self.dxs = [ opti.variable(model.ndx) for model in self.runningModels+[self.terminalModel] ]   
        self.acs = [ opti.variable(model.nv) for model in self.runningModels ]
        self.us =  [ opti.variable(model.nu) for model in self.runningModels ]
        self.xs =  [ m.integrate(x0,dx) for m,dx in zip(self.runningModels+[self.terminalModel],self.dxs) ]
        self.fs =  []
        for m in self.runningModels:     
            f_tmp = [opti.variable(3) for _ in range(len(m.contactIds)) ]
            self.fs.append(f_tmp)
        self.fs = self.fs

        self.cost, eq_constraints = self.make_ocp()

        opti.minimize(self.cost)

        opti.subject_to(eq_constraints == 0)

        p_opts = {}
        s_opts = {"tol": 1e-3,
            "acceptable_tol":1e-3,
            #"max_iter": 21,
            #"compl_inf_tol": 1e-2,
            #"constr_viol_tol": 1e-2
            #"resto_failure_feasibility_threshold": 1
            #"linear_solver": "ma57"
            }

        opti.solver("ipopt",p_opts,
                    s_opts)

        self.warmstart(x0, guess)

        ### SOLVE
        opti.solve_limited()
        for data in self.datas:
            data.x = self.opti.value(data.x)
            data.u = self.opti.value(data.u)
            data.xnext = self.opti.value(data.xnext)
            data.cost = self.opti.value(data.cost)
    
    def solve(self, x0, guess={}):
        """ Solve the NLP
        :param guess: ditctionary in the form: {'xs':array([T+1, n_states]), 'acs':array([T, nv]),\
            'us':array([T, nu]), 'fs': [array([T, nv]),]}
        """

        self.runningModels = [ self.casadiActionModels[self.problemData.contactSequence[t]] for t in range(self.problemData.T) ]
        self.terminalModel = self.casadiActionModels[self.problemData.contactSequence[self.problemData.T]]
        self.datas = [ModelData() for _ in range(self.problemData.T)]

        start_time = time() 

        self.use_ipopt_solver(x0, guess)

        self.iterationTime = time() - start_time
