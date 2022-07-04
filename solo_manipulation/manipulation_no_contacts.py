#from ocp import SimpleManipulationProblem
from time import time, sleep
from Controller import SimulationData, Controller
from ProblemData import ProblemData, ProblemDataFull
from Target import Target
from utils.BulletWrapper import BulletWrapper
from pinocchio.visualize import GepettoVisualizer
import numpy as np
from plot_utils import plot_mpc

def control_loop(init_guess, target):
    for t in range(horizon):
        print( "\nSTEP: ", str(t), "\n")
        m = sim.read_state()

        target.update(t)
        if t == 0:
            start_time = time()
            ctrl.compute_step(pd.x0)
            print("Time: ", time()-start_time, '\n')
            sim.send_torques(ctrl.results.x, ctrl.results.u, ctrl.results.k)
        else:
            target.shift_gait()
            start_time = time()
            ctrl.compute_step(m['x_m'], loadPreviousSol=True)
            print("Time: ", time()-start_time, '\n')
            sim.send_torques(ctrl.results.x, ctrl.results.u, ctrl.results.k)


if __name__ == "__main__":
    pd = ProblemDataFull() # Remember to modify also the Example Robot Data
    target = Target(pd)

    horizon = 100

    #device = Init_simulation(pd.x0[: pd.nq])
    ctrl = Controller(pd, target, 'crocoddyl')
    sim = BulletWrapper(ctrl)

    sim.store_measures()
    control_loop(None, target)
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


