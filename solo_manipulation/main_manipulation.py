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
        target.shift_gait()
        if t == 0:  
            ctrl.compute_step(pd.x0, guess=init_guess)
            sim.send_torques(ctrl.results.x, ctrl.results.u, ctrl.results.k)
        else:
            start_time = time()
            #ctrl.compute_step(ctrl.results.ocp_storage['xs'][-1][1], loadPreviousSol=True)
            ctrl.compute_step(m['x_m'], loadPreviousSol=True)
            print("Time: ", time()-start_time, '\n')
            sim.send_torques(ctrl.results.x, ctrl.results.u, ctrl.results.k)


if __name__ == "__main__":
    pd = ProblemData() # Remember to modify also the Example Robot Data
    target = Target(pd)

    horizon = 200

    ctrl = Controller(pd, target, 'crocoddyl')
    sim = BulletWrapper(ctrl)

    try:
        guesses = np.load('/tmp/sol_crocoddyl.npy', allow_pickle=True).item()
        init_guess = {'xs': list(guesses['xs']), 'us': list(guesses['us'])}
    except:
        init_guess = None
        print("No first guess")

    sim.store_measures()
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
    #viz.play(ctrl.get_q_sim_mpc().T, pd.dt_sim)


