import os
from time import time

import casadi
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from ProblemData import ProblemData
from Target import Target


plt.style.use('seaborn')
path = os.getcwd()


class ShootingNode():
    def __init__(self, pd: ProblemData, contactIds):

        self.pd = pd
        self.contactIds = contactIds
        self.freeIds = []

        cmodel = cpin.Model(pd.model)
        cdata = cmodel.createData()

        self.baseID = pd.model.getFrameId('base_link')
        pin.framesForwardKinematics(pd.model, pd.rdata, pd.q0)
        self.robotweight = - \
            sum([Y.mass for Y in pd.model.inertias]) * \
            pd.model.gravity.linear[2]
        [self.freeIds.append(idf)
         for idf in pd.allContactIds if idf not in contactIds]

        cx = casadi.SX.sym("x", pd.nx, 1)
        cq = casadi.SX.sym("q", pd.nq, 1)
        cv = casadi.SX.sym("v", pd.nv, 1)
        cx2 = casadi.SX.sym("x2", pd.nx, 1)
        cu = casadi.SX.sym("u", pd.nu, 1)
        ca = casadi.SX.sym("a", pd.nv, 1)
        ctau = casadi.SX.sym("tau", pd.ntau, 1)
        cdx = casadi.SX.sym("dx", pd.ndx, 1)
        cfs = [casadi.SX.sym("f"+cmodel.frames[idf].name, 3, 1) for idf in self.contactIds]
        R = casadi.SX.sym('R', 3, 3)
        R_ref = casadi.SX.sym('R_ref', 3, 3)

        # Build force list for ABA
        forces = [cpin.Force.Zero() for _ in cmodel.joints]
        # I am supposing here that all contact frames are on separate joints. This is asserted below:
        assert(len(set([cmodel.frames[idf].parentJoint for idf in contactIds])) == len(contactIds))
        for f, idf in zip(cfs, self.contactIds):
            # Contact forces introduced in ABA as spatial forces at joint frame. F (opti variable) is defined at contact frame
            forces[cmodel.frames[idf].parentJoint] = cmodel.frames[idf].placement * cpin.Force(f, 0*f)
        self.forces = cpin.StdVec_Force()
        for f in forces:
            self.forces.append(f)

        acc = cpin.aba(cmodel, cdata, cx[:pd.nq], cx[pd.nq:], ctau, self.forces)

        # Casadi Functions to wrap pinocchio methods (needed cause pinocchio.casadi methods can't handle MX variables used in potimization)
        # acceleration(x,u,f)  = ABA(q,v,tau,f) with x=q,v, tau=u, and f built using StdVec_Force syntaxt
        self.acc = casadi.Function('acc', [cx, ctau]+cfs, [acc])

        # integrate(x,dx) =   [q+dq,v+dv],   with the q+dq function implemented with pin.integrate.
        self.integrate = casadi.Function('xplus', [cx, cdx], [casadi.vertcat(cpin.integrate(cmodel, cx[:pd.nq], cdx[:pd.nv]),
                                                                                            cx[-pd.nv:]+cdx[-pd.nv:])])

        # integrate_q(q,dq) = pin.integrate(q,dq)
        self.integrate_q = casadi.Function('qplus', [cq, cv], [cpin.integrate(cmodel, cq, cv)])

        # Lie difference(x1,x2) = [ pin.difference(q1,q2),v2-v1 ]
        self.difference = casadi.Function('xminus', [cx, cx2], [casadi.vertcat(cpin.difference(cmodel, cx2[:pd.nq], cx[:pd.nq]),
                                                                                                cx2[pd.nq:]-cx[pd.nq:])])

        vel_reference_frame = pin.LOCAL #Reference frame in which velocities of base will be expressed (for locomotion)

        cpin.forwardKinematics(cmodel, cdata, cx[:pd.nq], cx[pd.nq:], ca)
        cpin.updateFramePlacements(cmodel, cdata)

        # com(x) = centerOfMass(x[:nq])
        self.com = casadi.Function('com', [cx], [cpin.centerOfMass(cmodel, cdata, cx[:pd.nq])])

        # Base link position and orientation
        self.baseTranslation = casadi.Function('base_translation', [cx], [cdata.oMf[self.baseID].translation])
        self.baseRotation = casadi.Function('base_rotation', [cx], [cdata.oMf[self.baseID].rotation])
        
        # Base velocity
        self.baseVelocityLin = casadi.Function('base_velocity_linear', [cx],
                                               [cpin.getFrameVelocity(cmodel, cdata, self.baseID, vel_reference_frame).linear])
        self.baseVelocityAng = casadi.Function('base_velocity_angular', [cx],
                                               [cpin.getFrameVelocity(cmodel, cdata, self.baseID, vel_reference_frame).angular])

        # feet[c](x) =  position of the foot <c> at configuration q=x[:nq]
        self.feet = {idf: casadi.Function('foot'+cmodel.frames[idf].name, [cx], [cdata.oMf[idf].translation]) for idf in pd.allContactIds}

        # Rfeet[c](x) =  orientation of the foot <c> at configuration q=x[:nq]
        self.Rfeet = {idf: casadi.Function('Rfoot'+cmodel.frames[idf].name, [cx], [cdata.oMf[idf].rotation]) for idf in pd.allContactIds}

        # vfeet[c](x) =  linear velocity of the foot <c> at configuration q=x[:nq] with vel v=x[nq:]
        self.vfeet = {idf: casadi.Function('vfoot'+cmodel.frames[idf].name,
                                           [cx], [cpin.getFrameVelocity(cmodel, cdata, idf, pin.LOCAL_WORLD_ALIGNED).linear])
                      for idf in pd.allContactIds}

        # vfeet[c](x,a) =  linear acceleration of the foot <c> at configuration q=x[:nq] with vel v=x[nq:] and acceleration a
        self.afeet = {idf: casadi.Function('afoot'+cmodel.frames[idf].name,
                                           [cx, ca], [cpin.getFrameClassicalAcceleration(cmodel, cdata, idf,
                                                                                         pin.LOCAL_WORLD_ALIGNED).linear])
                      for idf in pd.allContactIds}

        # wrapper for the inverse of the exponential map
        self.log3 = casadi.Function('log', [R, R_ref], [cpin.log3(R.T @ R_ref)])

    def init(self, data, x, a=None, u=None, fs=None, isTerminal=False):
        self.data = data
        self.data.x = x
        self.data.a = a
        self.data.u = u
        self.data.f = fs

        self.x = x
        self.isTerminal = isTerminal
        if not isTerminal:
            self.a = a
            self.u = u
            self.tau = casadi.vertcat(np.zeros(6), u)
            self.fs = fs

    def calc(self, x_ref, u_ref=None, target={}):
        '''
        This function return xnext,cost
        '''

        dt = self.pd.dt

        # Euler integration, using directly the acceleration <a> introduced as a slack variable.
        use_rk2 = False
        if not use_rk2:
            vnext = self.x[self.pd.nq:] + self.a*dt
            qnext = self.integrate_q(self.x[:self.pd.nq], vnext*dt)
        else:
            # half-dt step over x=(q,v)
            vm = self.x[self.pd.nq:] + self.a*.5*dt
            qm = self.integrate_q(self.x[:self.pd.nq], .5 * self.x[self.pd.nq:]*dt)
            xm = casadi.vertcat(qm, vm)
            amid = self.acc(xm, self.tau, *self.fs)

            # then simple Euler step over (qm, vm)
            qnext = self.integrate_q(qm, vm*dt)
            vnext = vm + amid*dt

        xnext = casadi.vertcat(qnext, vnext)
        self.data.xnext = xnext

        # Cost functions:
        self.compute_cost(x_ref, u_ref, target)
        self.data.cost = self.cost

        return xnext, self.cost

    def constraint_landing_feet_eq(self, x_ref):
        eq = []
        for stFoot in self.contactIds:
            eq.append(self.feet[stFoot](self.x)[2] - self.feet[stFoot](x_ref)[2])
            eq.append(self.vfeet[stFoot](self.x))
        return(casadi.vertcat(*eq))

    def constraint_standing_feet_eq(self):
        eq = []
        for stFoot in self.contactIds:
            eq.append(self.afeet[stFoot](self.x, self.a))  # stiff contact

        return(casadi.vertcat(*eq))

    def constraint_standing_feet_cost(self):
        # Friction cone
        for i, stFoot in enumerate(self.contactIds):
            R = self.Rfeet[stFoot](self.x)
            f_ = self.fs[i]
            fw = f_  # PREMULTIPLY BY R !!!!!

            A = np.matrix([[1, 0, -self.pd.mu], [-1, 0, -self.pd.mu], [6.1234234 *1e-17, 1, -self.pd.mu],\
                           [-6.1234234*1e-17, -1, -self.pd.mu], [0, 0, 1]])
            lb = np.array([-casadi.inf, -casadi.inf, -casadi.inf, -casadi.inf, 0])
            ub = np.array([0, 0, 0, 0, casadi.inf])
            r = A @ fw
            self.cost += self.pd.friction_cone_w * casadi.sumsqr(casadi.if_else(r <= lb, r, 0))/2 * self.pd.dt
            self.cost += self.pd.friction_cone_w * casadi.sumsqr(casadi.if_else(r >= ub, r, 0))/2 * self.pd.dt

    def constraint_control_cost(self):
        self.cost += self.pd.control_bound_w * casadi.sumsqr(casadi.if_else(self.u <= -self.pd.effort_limit, self.u, 0))/2 * self.pd.dt
        self.cost += self.pd.control_bound_w * casadi.sumsqr(casadi.if_else(self.u >= self.pd.effort_limit, self.u, 0))/2 * self.pd.dt

    def constraint_dynamics_eq(self):
        eq = []
        eq.append(self.acc(self.x, self.tau, *self.fs) - self.a)
        return(casadi.vertcat(*eq))

    def constraint_swing_feet_ineq(self, x_ref):
        ineq = []
        for sw_foot in self.freeIds:
            ineq.append(-self.feet[sw_foot](self.x)
                        [2] + self.feet[sw_foot](x_ref)[2])
        return(casadi.vertcat(*ineq))

    def force_reg_cost(self):
        for i, stFoot in enumerate(self.contactIds):
            R = self.Rfeet[stFoot](self.x)
            f_ = self.fs[i]
            fw = R @ f_
            self.cost += self.pd.force_reg_w * casadi.norm_2(fw[2] -
                                                             self.robotweight/len(self.contactIds)) * self.pd.dt

    def control_cost(self, u_ref):
        self.cost += 1/2*self.pd.control_reg_w * \
            casadi.sumsqr(self.u - u_ref) * self.pd.dt

    def body_reg_cost(self, x_ref):
        if self.isTerminal:
            self.cost += 1/2 * \
                casadi.sumsqr(self.pd.state_reg_w *
                              self.difference(self.x, x_ref))
            self.cost += 1/2 * \
                casadi.sumsqr(self.pd.terminal_velocity_w *
                              self.difference(self.x, x_ref))
        else:
            self.cost += 1/2 * \
                casadi.sumsqr(self.pd.state_reg_w *
                              self.difference(self.x, x_ref)) * self.pd.dt

    def target_cost(self, target):
        # I am Assuming just FR FOOt to be free
        for sw_foot in self.freeIds:

            self.cost += 1/2 * self.pd.foot_tracking_w * casadi.sumsqr(self.feet[sw_foot](self.x) - target[sw_foot]) * self.pd.dt

    def compute_cost(self, x_ref, u_ref, target):
        self.cost = 0
        if not self.isTerminal:
            self.constraint_standing_feet_cost()
            self.control_cost(u_ref)
            self.constraint_control_cost()
            self.target_cost(target)

        self.body_reg_cost(x_ref=x_ref)

        return self.cost
