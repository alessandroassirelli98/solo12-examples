import crocoddyl
import pinocchio
import numpy as np
from . import parameters_conf as conf


class SimpleManipulationProblem:
    def __init__(self, rmodel, lfFoot, rfFoot, lhFoot, rhFoot, integrator='euler', control='zero'):
            """ Construct quadrupedal-gait problem.

            :param rmodel: robot model
            :param lfFoot: name of the left-front foot
            :param rfFoot: name of the right-front foot
            :param lhFoot: name of the left-hind foot
            :param rhFoot: name of the right-hind foot
            :param integrator: type of the integrator (options are: 'euler', and 'rk4')
            :param control: type of control parametrization (options are: 'zero', 'one', and 'rk4')
            """
            self.rmodel = rmodel
            self.rdata = rmodel.createData()
            self.state = crocoddyl.StateMultibody(self.rmodel)
            self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
            # Getting the frame id for all the legs
            self.lfFootId = self.rmodel.getFrameId(lfFoot)
            self.rfFootId = self.rmodel.getFrameId(rfFoot)
            self.lhFootId = self.rmodel.getFrameId(lhFoot)
            self.rhFootId = self.rmodel.getFrameId(rhFoot)

            self.allContactsIds = (self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId)

            self.integrator = integrator
            if control == 'one':
                self.control = crocoddyl.ControlParametrizationModelPolyOne(self.actuation.nu)
            elif control == 'rk4':
                self.control = crocoddyl.ControlParametrizationModelPolyTwoRK(self.actuation.nu, crocoddyl.RKType.four)
            elif control == 'rk3':
                self.control = crocoddyl.ControlParametrizationModelPolyTwoRK(self.actuation.nu, crocoddyl.RKType.three)
            else:
                self.control = crocoddyl.ControlParametrizationModelPolyZero(self.actuation.nu)
            # Defining default state
            q0 = self.rmodel.referenceConfigurations["standing"]
            self.rmodel.defaultState = np.concatenate([q0, np.zeros(self.rmodel.nv)])
            self.firstStep = True
            # Defining the friction coefficient and normal
            self.mu = 0.7
            self.Rsurf = np.eye(3)

    def patternToId(self, gait):
        return [self.allContactsIds[i] for i,c in enumerate(gait) if c==1 ]

    def createMovingFootProblem(self, x0, x_ref, u_ref, gait, target, timeStep,):
        """ Create a shooting problem for a simple walking gait.

        :param x0: initial state
        """
        # Compute the current foot positions
        q0 = x0[:self.state.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2].item()
        self.u_ref = u_ref
        self.x_ref = x_ref
        
        freeIds = []
        contactSequence = [ self.patternToId(p) for p in gait ]

        self.T = T = len(gait)-1
        model = []
        
        for t in range(T):
            
            freeIds = [idf for idf in self.allContactsIds if idf not in contactSequence[t]]
            print("Contacts on ", contactSequence[t])
            print("Free on: ", freeIds)

            model += self.createFootstepModels(comRef, target[t], timeStep, contactSequence[t], freeIds, False)
        
        model += self.createFootstepModels(comRef, target[T], timeStep, contactSequence[T], freeIds, True)

        problem = crocoddyl.ShootingProblem(x0, model[:-1], model[-1])
        return problem

    def createFootstepModels(self, comPos0, target, timeStep, supportFootIds,
                             swingFootIds, isTerminal=False):
        """ Action models for a footstep phase.

        :param comPos0, initial CoM position
        :param feetPos0: initial position of the swinging feet
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: time step
        :param numKnots: number of knots for the footstep phase
        :param supportFootIds: Ids of the supporting feet
        :param swingFootIds: Ids of the swinging foot
        :return footstep action models
        """
        numLegs = len(supportFootIds) + len(swingFootIds)

        # Action models for the foot swing
        footSwingModel = []
        swingFootTask = []
        for i in swingFootIds:
            tref = target
            swingFootTask += [[i, pinocchio.SE3(np.eye(3), tref)]]

        comTask = comPos0
        footSwingModel += [
            self.createSwingFootModel(timeStep, supportFootIds, comTask=comTask, swingFootTask=swingFootTask, isTerminal=isTerminal)
        ]
        return footSwingModel

    def createSwingFootModel(self, timeStep, supportFootIds, comTask=None, swingFootTask=None, isTerminal=False):
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
                cone = crocoddyl.FrictionCone(self.Rsurf, self.mu, 4, False)
                coneResidual = crocoddyl.ResidualModelContactFrictionCone(self.state, i, cone, nu)
                #forceResidual = crocoddyl.ResidualModelContactForce(self.state, i, np.array([0,0,4]), 3, nu)
                coneActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
                #forceActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(-1000, 0))
                frictionCone = crocoddyl.CostModelResidual(self.state, coneActivation, coneResidual)
                #force = crocoddyl.CostModelResidual(self.state, forceActivation, forceResidual)
                costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, conf.friction_cone_w)
                #costModel.addCost(self.rmodel.frames[i].name + "_unilateral", force, 1e5)
            if swingFootTask is not None:
                for i in swingFootTask:
                    frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state, i[0], i[1].translation,
                                                                                    nu)
                    footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
                    costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, conf.foot_tracking_w)

            ctrlResidual = crocoddyl.ResidualModelControl(self.state, self.u_ref)    
            ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual) 
            costModel.addCost("ctrlReg", ctrlReg, conf.control_reg_w)

        stateResidual = crocoddyl.ResidualModelState(self.state, self.x_ref , nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(conf.state_reg_w**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        costModel.addCost("stateReg", stateReg, 1)

        """ lb = np.concatenate([self.state.lb[1:self.state.nv + 1], self.state.lb[-self.state.nv:]])
        ub = np.concatenate([self.state.ub[1:self.state.nv + 1], self.state.ub[-self.state.nv:]])
        stateBoundsResidual = crocoddyl.ResidualModelState(self.state, nu)
        stateBoundsActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub))
        stateBounds = crocoddyl.CostModelResidual(self.state, stateBoundsActivation, stateBoundsResidual)
        costModel.addCost("stateBounds", stateBounds, 1e3) """

        """ if(isTerminal):
            baseId = self.rmodel.getFrameId('base_link')
            terminalVelocityResidual = crocoddyl.ResidualModelFrameVelocity(self.state, baseId, 
                                                                    pinocchio.Motion.Zero(), pinocchio.WORLD, nu)
            terminalVelocityCost = crocoddyl.CostModelResidual(self.state, terminalVelocityResidual)
            costModel.addCost("terminalCost", terminalVelocityCost, conf.terminal_velocity_w) """

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)
        if self.integrator == 'euler':
            model = crocoddyl.IntegratedActionModelEuler(dmodel, self.control, timeStep)
        elif self.integrator == 'rk4':
            model = crocoddyl.IntegratedActionModelRK(dmodel, self.control, crocoddyl.RKType.four, timeStep)
        elif self.integrator == 'rk3':
            model = crocoddyl.IntegratedActionModelRK(dmodel, self.control, crocoddyl.RKType.three, timeStep)
        elif self.integrator == 'rk2':
            model = crocoddyl.IntegratedActionModelRK(dmodel, self.control, crocoddyl.RKType.two, timeStep)
        else:
            model = crocoddyl.IntegratedActionModelEuler(dmodel, self.control, timeStep)
        return model

'''
    def createFootSwitchModel(self, supportFootIds, swingFootTask, pseudoImpulse=False):
        """ Action model for a foot switch phase.

        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :param pseudoImpulse: true for pseudo-impulse models, otherwise it uses the impulse model
        :return action model for a foot switch phase
        """
        if pseudoImpulse:
            return self.createPseudoImpulseModel(supportFootIds, swingFootTask)
        else:
            return self.createImpulseModel(supportFootIds, swingFootTask)

    def createPseudoImpulseModel(self, supportFootIds, swingFootTask):
        """ Action model for pseudo-impulse models.

        A pseudo-impulse model consists of adding high-penalty cost for the contact velocities.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return pseudo-impulse differential action model
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
        for i in supportFootIds:
            cone = crocoddyl.FrictionCone(self.Rsurf, self.mu, 4, False)
            coneResidual = crocoddyl.ResidualModelContactFrictionCone(self.state, i, cone, nu)
            coneActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
            frictionCone = crocoddyl.CostModelResidual(self.state, coneActivation, coneResidual)
            costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e1)
        if swingFootTask is not None:
            for i in swingFootTask:
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state, i[0], i[1].translation,
                                                                                   nu)
                frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(self.state, i[0], pinocchio.Motion.Zero(),
                                                                             pinocchio.LOCAL, nu)
                footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
                impulseFootVelCost = crocoddyl.CostModelResidual(self.state, frameVelocityResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e7)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_impulseVel", impulseFootVelCost, 1e6)

        stateWeights = np.array([1.] * 3 + [500.] * 3 + [0.01] * (self.rmodel.nv - 6) + [10.] * self.rmodel.nv)
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-3)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)
        if self.integrator == 'euler':
            model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.)
        elif self.integrator == 'rk4':
            model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType.four, 0.)
        elif self.integrator == 'rk3':
            model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType.three, 0.)
        elif self.integrator == 'rk2':
            model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType.two, 0.)
        else:
            model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.)
        return model

    def createImpulseModel(self, supportFootIds, swingFootTask, JMinvJt_damping=1e-12, r_coeff=0.0):
        """ Action model for impulse models.

        An impulse model consists of describing the impulse dynamics against a set of contacts.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return impulse action model
        """
        # Creating a 3D multi-contact model, and then including the supporting foot
        impulseModel = crocoddyl.ImpulseModelMultiple(self.state)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ImpulseModel3D(self.state, i)
            impulseModel.addImpulse(self.rmodel.frames[i].name + "_impulse", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, 0)
        if swingFootTask is not None:
            for i in swingFootTask:
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state, i[0], i[1].translation,
                                                                                   0)
                footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e3)

        stateWeights = np.array([0] * 6 + [10.] * (self.rmodel.nv - 6) + [10.] * self.rmodel.nv)
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, 0)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        costModel.addCost("stateReg", stateReg, 1e1)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        model = crocoddyl.ActionModelImpulseFwdDynamics(self.state, impulseModel, costModel)
        model.JMinvJt_damping = JMinvJt_damping
        model.r_coeff = r_coeff
        return model        # Action model for the foot switch
        #footSwitchModel = self.createFootSwitchModel(supportFootIds, swingFootTask)

'''

def plotSolution(solver, bounds=True, figIndex=1, figTitle="", show=True):
    import matplotlib.pyplot as plt
    xs, us = [], []
    if bounds:
        us_lb, us_ub = [], []
        xs_lb, xs_ub = [], []
    if isinstance(solver, list):
        rmodel = solver[0].problem.runningModels[0].state.pinocchio
        for s in solver:
            xs.extend(s.xs[:-1])
            us.extend(s.us)
            if bounds:
                models = s.problem.runningModels.tolist() + [s.problem.terminalModel]
                for m in models:
                    us_lb += [m.u_lb]
                    us_ub += [m.u_ub]
                    xs_lb += [m.state.lb]
                    xs_ub += [m.state.ub]
    else:
        rmodel = solver.problem.runningModels[0].state.pinocchio
        xs, us = solver.xs, solver.us
        if bounds:
            models = solver.problem.runningModels.tolist() + [solver.problem.terminalModel]
            for m in models:
                us_lb += [m.u_lb]
                us_ub += [m.u_ub]
                xs_lb += [m.state.lb]
                xs_ub += [m.state.ub]

    # Getting the state and control trajectories
    nx, nq, nu = xs[0].shape[0], rmodel.nq, us[0].shape[0]
    X = [0.] * nx
    U = [0.] * nu
    if bounds:
        U_LB = [0.] * nu
        U_UB = [0.] * nu
        X_LB = [0.] * nx
        X_UB = [0.] * nx
    for i in range(nx):
        X[i] = [np.asscalar(x[i]) for x in xs]
        if bounds:
            X_LB[i] = [np.asscalar(x[i]) for x in xs_lb]
            X_UB[i] = [np.asscalar(x[i]) for x in xs_ub]
    for i in range(nu):
        U[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else 0 for u in us]
        if bounds:
            U_LB[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else np.nan for u in us_lb]
            U_UB[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else np.nan for u in us_ub]

    # Plotting the joint positions, velocities and torques
    plt.figure(figIndex)
    plt.suptitle(figTitle)
    legJointNames = ['HAA', 'HFE', 'KFE']
    # LF foot
    plt.subplot(4, 3, 1)
    plt.title('joint position [rad]')
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(7, 10))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(7, 10))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(7, 10))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(4, 3, 2)
    plt.title('joint velocity [rad/s]')
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 6, nq + 9))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 6, nq + 9))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 6, nq + 9))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(4, 3, 3)
    plt.title('joint torque [Nm]')
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(0, 3))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(0, 3))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(0, 3))]
    plt.ylabel('LF')
    plt.legend()

    # LH foot
    plt.subplot(4, 3, 4)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(10, 13))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(10, 13))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(10, 13))]
    plt.ylabel('LH')
    plt.legend()
    plt.subplot(4, 3, 5)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 9, nq + 12))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 9, nq + 12))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 9, nq + 12))]
    plt.ylabel('LH')
    plt.legend()
    plt.subplot(4, 3, 6)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(3, 6))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(3, 6))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(3, 6))]
    plt.ylabel('LH')
    plt.legend()

    # RF foot
    plt.subplot(4, 3, 7)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(13, 16))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(13, 16))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(13, 16))]
    plt.ylabel('RF')
    plt.legend()
    plt.subplot(4, 3, 8)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 12, nq + 15))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 12, nq + 15))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 12, nq + 15))]
    plt.ylabel('RF')
    plt.legend()
    plt.subplot(4, 3, 9)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(6, 9))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(6, 9))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(6, 9))]
    plt.ylabel('RF')
    plt.legend()

    # RH foot
    plt.subplot(4, 3, 10)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(16, 19))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(16, 19))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(16, 19))]
    plt.ylabel('RH')
    plt.xlabel('knots')
    plt.legend()
    plt.subplot(4, 3, 11)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 15, nq + 18))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 15, nq + 18))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 15, nq + 18))]
    plt.ylabel('RH')
    plt.xlabel('knots')
    plt.legend()
    plt.subplot(4, 3, 12)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(9, 12))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(9, 12))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(9, 12))]
    plt.ylabel('RH')
    plt.legend()
    plt.xlabel('knots')

    plt.figure(figIndex + 1)
    plt.suptitle(figTitle)
    rdata = rmodel.createData()
    Cx = []
    Cy = []
    for x in xs:
        q = x[:nq]
        c = pinocchio.centerOfMass(rmodel, rdata, q)
        Cx.append(np.asscalar(c[0]))
        Cy.append(np.asscalar(c[1]))
    plt.plot(Cx, Cy)
    plt.title('CoM position')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    if show:
        plt.show()
