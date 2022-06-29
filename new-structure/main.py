#from ocp import SimpleManipulationProblem
from Controller import Controller
from ProblemData import ProblemData, Target
from utils.PyBulletSimulator import PyBulletSimulator
from pinocchio.visualize import GepettoVisualizer
import numpy as np



def control_loop(init_guess, target):
    for t in range(horizon):
        m = ctrl.read_state()

        target.update(t)
        if t == 0:
            ctrl.compute_step(pd.x0, init_guess)
        else:
            pd.shift_gait()
            ctrl.compute_step(pd.x0)


if __name__ == "__main__":
    pd = ProblemData()
    target = Target(pd)

    horizon = 1
    dt_ocp = pd.dt
    dt_sim = 0.001
    r = int(dt_ocp/dt_sim)

    #device = Init_simulation(pd.x0[: pd.nq])
    ctrl = Controller(pd, target, dt_sim, r, 'ipopt')

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
