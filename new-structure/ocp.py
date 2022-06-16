from problem import ProblemData
from data import Data
import crocoddyl
import pinocchio as pin
import numpy as np

class SimpleManipulationProblem:
    def __init__(self, solverType, problemData):
        self.solverType = solverType
        self.problem_data = problemData

        self.results = Data()

        self.rmodel = self.problem_data.robot.model
        self.rdata = self.rmodel.createData()
        if solverType == "crocoddyl":
            self.state = crocoddyl.StateMultibody(self.rmodel)
            self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)

        # Getting the frame id for all the feet
        self.lfFootId = self.rmodel.getFrameId(self.problem_data.lfFoot)
        self.rfFootId = self.rmodel.getFrameId(self.problem_data.rfFoot)
        self.lhFootId = self.rmodel.getFrameId(self.problem_data.lhFoot)
        self.rhFootId = self.rmodel.getFrameId(self.problem_data.rhFoot)
        self.allContactsIds = (self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId)

        self.contactSequence = [ self.patternToId(p) for p in self.problem_data.gait ]
        self.T = T = len(self.problem_data.gait)-1

    def patternToId(self, gait):
        return [self.allContactsIds[i] for i,c in enumerate(gait) if c==1 ]

### Crocoddyl side
    def make_crocoddyl_ocp(self, x0, x_ref, u_ref):
        """ Create a shooting problem for a simple walking gait.

        :param x0: initial state
        """
        self.control = crocoddyl.ControlParametrizationModelPolyZero(self.actuation.nu)
        self.x_ref = x_ref
        self.u_ref = u_ref

        # Compute the current foot positions
        q0 = x0[:self.state.nq]
        pin.forwardKinematics(self.rmodel, self.rdata, q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        
        model = []
        for t in range(self.T):
            target = self.problem_data.target
            freeIds = [idf for idf in self.allContactsIds if idf not in self.contactSequence[t]]
            contactIds = self.contactSequence[t]
            model += self.createFootstepModels(target[t], contactIds, freeIds) 
        model += self.createFootstepModels(target[t], contactIds, freeIds, True)

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
                cone = crocoddyl.FrictionCone(self.problem_data.Rsurf, self.problem_data.mu, 4, False)
                coneResidual = crocoddyl.ResidualModelContactFrictionCone(self.state, i, cone, nu)
                #forceResidual = crocoddyl.ResidualModelContactForce(self.state, i, np.array([0,0,4]), 3, nu)
                coneActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
                #forceActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(-1000, 0))
                frictionCone = crocoddyl.CostModelResidual(self.state, coneActivation, coneResidual)
                #force = crocoddyl.CostModelResidual(self.state, forceActivation, forceResidual)
                costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, self.problem_data.friction_cone_w)
                #costModel.addCost(self.rmodel.frames[i].name + "_unilateral", force, 1e5)
            if swingFootTask is not None:
                for i in swingFootTask:
                    frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state, i[0], i[1].translation,
                                                                                    nu)
                    footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
                    costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, self.problem_data.foot_tracking_w)

            ctrlResidual = crocoddyl.ResidualModelControl(self.state, self.u_ref)    
            ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual) 
            costModel.addCost("ctrlReg", ctrlReg, self.problem_data.control_reg_w)

            stateResidual = crocoddyl.ResidualModelState(self.state, self.x_ref , nu)
            stateActivation = crocoddyl.ActivationModelWeightedQuad(self.problem_data.state_reg_w**2)
            stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
            costModel.addCost("stateReg", stateReg, 1)

        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)

        model = crocoddyl.IntegratedActionModelEuler(dmodel, self.control, self.problem_data.dt)

        return model

### IPOPT side

# Solve
    def solve(self, x0, x_ref, u_ref):
        if self.solverType == "crocoddyl":
            problem = self.make_crocoddyl_ocp(x0, x_ref, u_ref)
            self._solver = crocoddyl.SolverDDP(problem)
            self._solver.setCallbacks([crocoddyl.CallbackVerbose()])

            xs = [x0] * (self._solver.problem.T + 1)
            us = self._solver.problem.quasiStatic([x0] * self._solver.problem.T)

            self._solver.solve(xs, us, 100, False, 0.1)

            self.results.ocp_storage['xs'] += [np.array(self._solver.xs.tolist())]
            self.results.ocp_storage['us'] += [np.array(self._solver.us.tolist())]
        
        
