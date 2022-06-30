#from ocp import SimpleManipulationProblem
from time import time, sleep
from Controller import SimulationData, Controller
from ProblemData import ProblemData, Target
from utils.PyBulletSimulator import PyBulletSimulator
from pinocchio.visualize import GepettoVisualizer
import numpy as np
from plot_utils import plot_mpc

def control_loop(init_guess, target):
    for t in range(horizon):
        m = ctrl.read_state()
        target.update(t)
        if t == 0:
            ctrl.compute_step(pd.x0, init_guess)
            ctrl.send_torques(ctrl.results.x, ctrl.results.u, ctrl.results.k)
        else:
            target.shift_gait()
            ctrl.compute_step(m['x_m'])
            ctrl.send_torques(ctrl.results.x, ctrl.results.u, ctrl.results.k)


if __name__ == "__main__":
    pd = ProblemData()
    target = Target(pd)

    horizon = 20

    #device = Init_simulation(pd.x0[: pd.nq])
    ctrl = Controller(pd, target, 'ipopt')

    guesses = np.load('/tmp/sol_crocoddyl.npy', allow_pickle=True).item()
    init_guess = {'xs': list(guesses['xs']), 'us': list(guesses['us']),
                  'acs': guesses['acs'], 'fs': guesses['fs']}
    control_loop(init_guess, target)
    ctrl.results.make_arrays()
    plot_mpc(ctrl)

    try:
        viz = GepettoVisualizer(
            pd.model, pd.robot.collision_model, pd.robot.visual_model)
        viz.initViewer()
        viz.loadViewerModel()
        gv = viz.viewer.gui
    except:
        print("No viewer")

    # SHOW OCP RESULT
    #viz.play(ctrl.results.ocp_storage['xs'][0][:, :19].T, pd.dt)
    viz.play(ctrl.get_q_mpc().T, pd.dt)
    sleep(1)
    viz.play(ctrl.get_q_sim_mpc().T, pd.dt_sim)


