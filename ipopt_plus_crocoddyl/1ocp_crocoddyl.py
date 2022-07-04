from utils.PyBulletSimulator import PyBulletSimulator
from utils.plot_utils import plot_ocp
import numpy as np
from Controller_crocoddyl import Controller, Results
from pinocchio.visualize import GepettoVisualizer
import matplotlib.pyplot as plt
import pybullet as p

horizon = 1
dt_ocp = 0.015
dt_sim = 0.001
r = int(dt_ocp/dt_sim)


def Init_simulation(q_init):
    device = PyBulletSimulator()
    device.Init(q_init, 0, True, True, dt_sim)
    return device

def tuple_to_array(tup):
    a = np.array([element for tupl in tup for element in tupl])
    return a

def control_loop(ctrl):

    ctrl.create_target(0)

    try:
        #guesses = np.load('/tmp/sol_ipopt.npy',allow_pickle=True).item()
        ws = {'xs': list(guesses['xs']), 'us': list(guesses['us']) }
        ctrl.compute_step(ctrl.x0, ctrl.x0, ctrl.u0, ws)
        print("Solving with warmstart")
    except:
        ctrl.compute_step(ctrl.x0, ctrl.x0, ctrl.u0)
        print("Solving without warmstart")



if __name__ == '__main__':
    ctrl = Controller(10, 50, dt_ocp)
    local_res = Results()

    control_loop(ctrl)

    local_res.x_m = np.array(local_res.x_m)

    try:
        viz = GepettoVisualizer(ctrl.solo.model,ctrl.solo.robot.collision_model, ctrl.solo.robot.visual_model)
        viz.initViewer()
        viz.loadViewerModel()
        gv = viz.viewer.gui
    except:
        print("No viewer"  )

    viz.play(ctrl.results.ocp_storage['xs'][1][:, :19].T, dt_ocp) # SHOW OCP RESULTS
    #viz.play(local_res.x_m[:, :19].T, dt_sim) # SHOW PYBULLET SIMULATION
    totalcost = sum( [c.cost for c in ctrl._solver.problem.runningDatas.tolist()] ) + ctrl._solver.problem.terminalData.cost
    print("Total cost is: ", totalcost)


    np.save(open('/tmp/sol_crocoddyl.npy', "wb"),
        {
            "xs": ctrl.results.ocp_storage['xs'][1],
            "acs": ctrl.get_croco_acc(),
            "us": ctrl.results.ocp_storage['us'][0],
            "fs": ctrl.get_croco_forces_ws()} )

    #plot_ocp(ctrl, ctrl.results, local_res, 0.001)
