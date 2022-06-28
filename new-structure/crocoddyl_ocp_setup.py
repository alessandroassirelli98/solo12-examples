from problem import ProblemData
import crocoddyl
import pinocchio as pin
import numpy as np

class SimpleManipulationProblem:
    def __init__(self, problemData):
        
        self.pd = problemData

        self.rmodel = self.pd.robot.model
        self.rdata = self.rmodel.createData()
        
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)

    def make_crocoddyl_ocp(self, x0):
        """ Create a shooting problem for a simple walking gait.

        :param x0: initial state
        """
        self.control = crocoddyl.ControlParametrizationModelPolyZero(self.actuation.nu)

        # Compute the current foot positions
        q0 = x0[:self.state.nq]
        pin.forwardKinematics(self.rmodel, self.rdata, q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        
        model = []
        for t in range(self.pd.T):
            target = self.pd.target
            freeIds = [idf for idf in self.pd.allContactIds if idf not in self.pd.contactSequence[t]]
            contactIds = self.pd.contactSequence[t]
            model += self.createFootstepModels(target[t], contactIds, freeIds) 

        freeIds = [idf for idf in self.pd.allContactIds if idf not in self.pd.contactSequence[self.pd.T]]
        contactIds = self.pd.contactSequence[self.pd.T]
        model += self.createFootstepModels(target[self.pd.T], contactIds, freeIds, True)

        problem = crocoddyl.ShootingProblem(x0, model[:-1], model[-1])

        return problem
    
    def createFootstepModels(self,  target, supportFootIds,
                             swingFootIds, isTerminal=False):
        """ Action models for a footstep phase.
        :param numKnots: number of knots for the footstep phase
        :param supportFootIds: Ids of the supporting feet
        :param swingFootIds: Ids of the swinging foot
        :return footstep action models
        """
        # Action models for the foot swing
        footSwingModel = []
        swingFootTask = []
        for i in swingFootIds:
            tref = target
            swingFootTask += [[i, pin.SE3(np.eye(3), tref)]]

        footSwingModel += [
            self.createSwingFootModel(supportFootIds, swingFootTask=swingFootTask, isTerminal=isTerminal)
        ]
        return footSwingModel

    def createSwingFootModel(self, supportFootIds, swingFootTask=None, isTerminal=False):
        """ Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        nu = self.actuation.nu
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel3D(self.state, i, np.array([0., 0., 0.]), nu,
                                                           np.array([0., 50.]))
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)

        if not isTerminal:

            for i in supportFootIds:
                cone = crocoddyl.FrictionCone(self.pd.Rsurf, self.pd.mu, 4, False)
                coneResidual = crocoddyl.ResidualModelContactFrictionCone(self.state, i, cone, nu)
                #forceResidual = crocoddyl.ResidualModelContactForce(self.state, i, np.array([0,0,4]), 3, nu)
                coneActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
                #forceActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(-1000, 0))
                frictionCone = crocoddyl.CostModelResidual(self.state, coneActivation, coneResidual)
                #force = crocoddyl.CostModelResidual(self.state, forceActivation, forceResidual)
                costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, self.pd.friction_cone_w)
                #costModel.addCost(self.rmodel.frames[i].name + "_unilateral", force, 1e5)
            if swingFootTask is not None:
                for i in swingFootTask:
                    frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state, i[0], i[1].translation,
                                                                                    nu)
                    footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
                    costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, self.pd.foot_tracking_w)

            ctrlResidual = crocoddyl.ResidualModelControl(self.state, self.pd.uref)    
            ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual) 
            costModel.addCost("ctrlReg", ctrlReg, self.pd.control_reg_w)

            stateResidual = crocoddyl.ResidualModelState(self.state, self.pd.xref , nu)
            stateActivation = crocoddyl.ActivationModelWeightedQuad(self.pd.state_reg_w**2)
            stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
            costModel.addCost("stateReg", stateReg, 1)

        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)

        model = crocoddyl.IntegratedActionModelEuler(dmodel, self.control, self.pd.dt)

        return model

# Solve
    def solve(self, x0, guess={}):
        problem = self.make_crocoddyl_ocp(x0)
        self._solver = crocoddyl.SolverDDP(problem)
        self._solver.setCallbacks([crocoddyl.CallbackVerbose()])

        for g in guess:
            if guess[g] == []:
                print("No warmstart provided")    
                xs = [x0] * (self._solver.problem.T + 1)
                us = self._solver.problem.quasiStatic([x0] * self._solver.problem.T)
                break
            else:
                xs = guess['xs']
                us = guess['us']

        self._solver.solve(xs, us, 100, False, 0.1)

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
    
    def get_results(self):
        x = self._solver.xs.tolist()
        a = self.get_croco_acc()
        u = self._solver.us.tolist()
        f_ws = self.get_croco_forces_ws()
        return None, x, a, u, f_ws, None
        
        
