#from ocp import SimpleManipulationProblem
from Controller import Controller
from ProblemData import ProblemData
from utils.PyBulletSimulator import PyBulletSimulator
from pinocchio.visualize import GepettoVisualizer
import numpy as np


class Target:
    def __init__(self, pd: ProblemData):
        self.FR_foot0 = np.array([0.1946, -0.16891, 0.017])
        self.A = np.array([0, 0.035, 0.035])
        self.offset = np.array([0.05, 0., 0.06])
        self.freq = np.array([0, 2.5, 2.5])
        self.phase = np.array([0, 0, np.pi/2])

        self.pd = pd

    def update(self, t):
        target = []
        for n in range(self.pd.n_nodes):
            target += [self.FR_foot0 + self.offset + self.A *
                       np.sin(2*np.pi*self.freq * (n+t) * self.pd.dt + self.phase)]
        self.pd.target = np.array(target)


def control_loop(init_guess, target):
    for t in range(horizon):
        m = ctrl.read_state()

        target.update(t)
        if t == 0:
            ctrl.compute_step(pd.x0, init_guess)
        else:
            pd.shift_gait()
            ctrl.compute_step(pd.x0)


if __name__ == "main"():
    pd = ProblemData()

    horizon = 1
    dt_ocp = pd.dt
    dt_sim = 0.001
    r = int(dt_ocp/dt_sim)

    #device = Init_simulation(pd.x0[: pd.nq])
    target = Target(pd)
    ctrl = Controller(pd, dt_sim, r, 'ipopt')

    guesses = np.load('/tmp/sol_crocoddyl.npy', allow_pickle=True).item()
    init_guess = {'xs': list(guesses['xs']), 'us': list(guesses['us']),
                  'acs': guesses['acs'], 'fs': guesses['fs']}
    control_loop(init_guess, target)

    try:
        viz = GepettoVisualizer(
            pd.model, pd.robot.collision_model, pd.robot.visual_model)
        viz.initViewer()
        viz.loadViewerModel()
        gv = viz.viewer.gui
    except:
        print("No viewer")

    # SHOW OCP RESULT
    viz.play(ctrl.results.ocp_storage['xs'][0][:, :19].T, pd.dt)
