from caracal.contact import ContactSchedule, ContactPhase, ContactType
from caracal.trajectory import SwingFootTrajectoryGenerator
import numpy as np

class QuadrupedalGaitGenerator:
    def __init__(self, rmodel, dt=1e-2, S=4, lf="LF_FOOT", lh="LH_FOOT", rf="RF_FOOT", rh="RH_FOOT"):
        self._dt = dt
        self._S = S
        self._contactNames = [lf, lh, rf, rh]
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.lfFootId = rmodel.getFrameId(lf)
        self.rfFootId = rmodel.getFrameId(rf)
        self.lhFootId = rmodel.getFrameId(lh)
        self.rhFootId = rmodel.getFrameId(rh)

    def stand(self, N_ds):
        gait = ContactSchedule(self._dt, N_ds, self._S, self._contactNames)
        lf, lh, rf, rh = self._contactNames
        gait.addSchedule(lh, [ContactPhase(N_ds)])
        gait.addSchedule(lf, [ContactPhase(N_ds)])
        gait.addSchedule(rh, [ContactPhase(N_ds)])
        gait.addSchedule(rf, [ContactPhase(N_ds)])
        return gait

    def walk(self, contacts, N_ds, N_ss, N_uds=0, N_uss=0, stepHeight=0.15, startPhase=True, endPhase=True):
        N_0 = 0
        if startPhase:
            N_0 = N_ds
        if endPhase:
            N = N_0 + 2 * N_ds + 4 * N_ss - 2 * N_uss - N_uds
        else:
            N = N_0 + N_ds + 4 * N_ss - 2 * N_uss - N_uds
        gait = ContactSchedule(self._dt, N, self._S, self._contactNames)
        lf, lh, rf, rh = self._contactNames
        lfSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][lf], contacts[1][lf])
        lhSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][lh], contacts[1][lh])
        rfSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][rf], contacts[1][rf])
        rhSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][rh], contacts[1][rh])
        gait.addSchedule(
            lh, [ContactPhase(N_0),
                 ContactPhase(N_ss, trajectory=lhSwingTraj),
                 ContactPhase(N - (N_0 + N_ss))])
        gait.addSchedule(lf, [
            ContactPhase(N_0 + N_ss - N_uss),
            ContactPhase(N_ss, trajectory=lfSwingTraj),
            ContactPhase(N - (N_0 + 2 * N_ss - N_uss))
        ])
        gait.addSchedule(rh, [
            ContactPhase(N_0 + N_ds + 2 * N_ss - N_uss - N_uds),
            ContactPhase(N_ss, trajectory=rhSwingTraj),
            ContactPhase(N - (N_0 + N_ds + 3 * N_ss - N_uss - N_uds))
        ])
        gait.addSchedule(rf, [
            ContactPhase(N_0 + N_ds + 3 * N_ss - 2 * N_uss - N_uds),
            ContactPhase(N_ss, trajectory=rfSwingTraj),
            ContactPhase(N - (N_0 + N_ds + 4 * N_ss - 2 * N_uss - N_uds))
        ])
        return gait

    def moveFoot(self, contacts, N_ds, N_ss, N_uds=0, N_uss=0, stepHeight=0.15, startPhase=True, endPhase=True):
        N_0 = 0
        if startPhase:
            N_0 = N_ds
        if endPhase:
            N = N_0 + N_ds + 2 * N_ss - N_uss
        else:
            N = N_0 + 2 * N_ss - N_uss
        gait = ContactSchedule(self._dt, N, self._S, self._contactNames)
        lf, lh, rf, rh = self._contactNames
        rfSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][rf], contacts[1][rf])
        
        gait.addSchedule(lh, [ContactPhase(N)])
        gait.addSchedule(
            rf, [ContactPhase(N_0),
                 ContactPhase(N_ss, trajectory=rfSwingTraj),
                 ContactPhase(N - (N_0 + N_ss))])
        gait.addSchedule(lf, [ContactPhase(N)])
        gait.addSchedule(rh, [ContactPhase(N)])

        return gait

    def trot(self, contacts, N_ds, N_ss, N_uss=0, N_uds=0, stepHeight=0.15, startPhase=True, endPhase=True):
        N_0 = 0
        if startPhase:
            N_0 = N_ds
        if endPhase:
            N = N_0 + N_ds + 2 * N_ss - N_uss
        else:
            N = N_0 + 2 * N_ss - N_uss
        gait = ContactSchedule(self._dt, N, self._S, self._contactNames)
        lf, lh, rf, rh = self._contactNames
        lfSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][lf], contacts[1][lf])
        lhSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][lh], contacts[1][lh])
        rfSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][rf], contacts[1][rf])
        rhSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][rh], contacts[1][rh])
        gait.addSchedule(
            lh, [ContactPhase(N_0),
                 ContactPhase(N_ss, trajectory=lhSwingTraj),
                 ContactPhase(N - (N_0 + N_ss))])
        gait.addSchedule(
            rf, [ContactPhase(N_0),
                 ContactPhase(N_ss, trajectory=rfSwingTraj),
                 ContactPhase(N - (N_0 + N_ss))])
        gait.addSchedule(lf, [
            ContactPhase(N_0 + N_ss - N_uss),
            ContactPhase(N_ss, trajectory=lfSwingTraj),
            ContactPhase(N - (N_0 + 2 * N_ss - N_uss))
        ])
        gait.addSchedule(rh, [
            ContactPhase(N_0 + N_ss - N_uss),
            ContactPhase(N_ss, trajectory=rhSwingTraj),
            ContactPhase(N - (N_0 + 2 * N_ss - N_uss))
        ])
        return gait

    def pace(self, contacts, N_ds, N_ss, N_uss=0, N_uds=0, stepHeight=0.15, startPhase=True, endPhase=True):
        N_0 = 0
        if startPhase:
            N_0 = N_ds
        if endPhase:
            N = N_0 + N_ds + 2 * N_ss - N_uss
        else:
            N = N_0 + 2 * N_ss - N_uss
        gait = ContactSchedule(self._dt, N, self._S, self._contactNames)
        lf, lh, rf, rh = self._contactNames
        lfSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][lf], contacts[1][lf])
        lhSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][lh], contacts[1][lh])
        rfSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][rf], contacts[1][rf])
        rhSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][rh], contacts[1][rh])
        gait.addSchedule(
            lf, [ContactPhase(N_0),
                 ContactPhase(N_ss, trajectory=lfSwingTraj),
                 ContactPhase(N - (N_0 + N_ss))])
        gait.addSchedule(
            lh, [ContactPhase(N_0),
                 ContactPhase(N_ss, trajectory=lhSwingTraj),
                 ContactPhase(N - (N_0 + N_ss))])
        gait.addSchedule(rf, [
            ContactPhase(N_0 + N_ss - N_uss),
            ContactPhase(N_ss, trajectory=rfSwingTraj),
            ContactPhase(N - (N_0 + 2 * N_ss - N_uss))
        ])
        gait.addSchedule(rh, [
            ContactPhase(N_0 + N_ss - N_uss),
            ContactPhase(N_ss, trajectory=rhSwingTraj),
            ContactPhase(N - (N_0 + 2 * N_ss - N_uss))
        ])
        return gait

    def bound(self, contacts, N_ds, N_ss, N_uss=0, N_uds=0, stepHeight=0.15, startPhase=True, endPhase=True):
        N_0 = 0
        if startPhase:
            N_0 = N_ds
        if endPhase:
            N = N_0 + N_ds + 2 * N_ss - N_uss
        else:
            N = N_0 + 2 * N_ss - N_uss
        gait = ContactSchedule(self._dt, N, self._S, self._contactNames)
        lf, lh, rf, rh = self._contactNames
        lfSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][lf], contacts[1][lf])
        lhSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][lh], contacts[1][lh])
        rfSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][rf], contacts[1][rf])
        rhSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][rh], contacts[1][rh])
        gait.addSchedule(
            lf, [ContactPhase(N_0),
                 ContactPhase(N_ss, trajectory=lfSwingTraj),
                 ContactPhase(N - (N_0 + N_ss))])
        gait.addSchedule(
            rf, [ContactPhase(N_0),
                 ContactPhase(N_ss, trajectory=rfSwingTraj),
                 ContactPhase(N - (N_0 + N_ss))])
        gait.addSchedule(lh, [
            ContactPhase(N_0 + N_ss - N_uss),
            ContactPhase(N_ss, trajectory=lhSwingTraj),
            ContactPhase(N - (N_0 + 2 * N_ss - N_uss))
        ])
        gait.addSchedule(rh, [
            ContactPhase(N_0 + N_ss - N_uss),
            ContactPhase(N_ss, trajectory=rhSwingTraj),
            ContactPhase(N - (N_0 + 2 * N_ss - N_uss))
        ])
        return gait

    def jump(self, contacts, N_ds, N_ss, N_uss=0, N_uds=0, stepHeight=0.15, startPhase=True, endPhase=True):
        N_0 = 0
        if startPhase:
            N_0 = N_ds
        if endPhase:
            N = N_0 + 2 * N_ss + N_ds
        else:
            N = N_0 + N_ss + N_ds
        gait = ContactSchedule(self._dt, N, self._S, self._contactNames)
        lf, lh, rf, rh = self._contactNames
        lfSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][lf], contacts[1][lf])
        lhSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][lh], contacts[1][lh])
        rfSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][rf], contacts[1][rf])
        rhSwingTraj = SwingFootTrajectoryGenerator(self._dt, N_ss, stepHeight, contacts[0][rh], contacts[1][rh])
        gait.addSchedule(lf, [
            ContactPhase(N_0 + N_ds),
            ContactPhase(N_ss, trajectory=lfSwingTraj),
            ContactPhase(N - (N_0 + N_ds + N_ss))
        ])
        gait.addSchedule(rf, [
            ContactPhase(N_0 + N_ds),
            ContactPhase(N_ss, trajectory=rfSwingTraj),
            ContactPhase(N - (N_0 + N_ds + N_ss))
        ])
        gait.addSchedule(lh, [
            ContactPhase(N_0 + N_ds),
            ContactPhase(N_ss, trajectory=lhSwingTraj),
            ContactPhase(N - (N_0 + N_ds + N_ss))
        ])
        gait.addSchedule(rh, [
            ContactPhase(N_0 + N_ds),
            ContactPhase(N_ss, trajectory=rhSwingTraj),
            ContactPhase(N - (N_0 + N_ds + N_ss))
        ])
        return gait

    def create_target(self, t_, foot_idx, stepNodes, timestep):
        # Here we create the target of the control
        # FR Foot moving in a circular way

        foot0 = self.rdata.oMf[foot_idx].translation
        A = np.array([0, 0.05, 0.05]) 
        offset = np.array([0.05, 0., 0.06])
        freq = np.array([0, 1.5, 1.5])
        phase = np.array([0,0,np.pi/2])

        target = []
        for t in range(stepNodes): target += [foot0 + offset +A*np.sin(2*np.pi*freq * (t+t_)* timestep + phase)]
        return target