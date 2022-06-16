from utils.PyBulletSimulator import PyBulletSimulator
from utils.plot_utils import plot_ocp
import numpy as np
from Controller_crocoddyl import Controller, Results
from pinocchio.visualize import GepettoVisualizer
import matplotlib.pyplot as plt
import pybullet as p

horizon = 25
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

def interpolate_traj(q_des, v_des):
    measures = read_state()
    qj_des_i = np.linspace(measures['qj_m'], q_des, r)
    vj_des_i = np.linspace(measures['vj_m'], v_des, r)

    return qj_des_i, vj_des_i

def read_state():
    device.parse_sensor_data()
    qj_m = device.joints.positions
    vj_m = device.joints.velocities
    bp_m = tuple_to_array(device.baseState)
    bv_m = tuple_to_array(device.baseVel)
    x_m = np.concatenate([bp_m, qj_m, bv_m, vj_m])

    return {'qj_m': qj_m, 'vj_m': vj_m, 'x_m': x_m}

def store_measures(all=True):
    m = read_state()
    local_res.x_m += [m['x_m']]
    if all == True:
        local_res.tau += [device.jointTorques]

def send_torques(ctrl):
    u = ctrl._solver.us[0]
    x = ctrl._solver.xs[1]
    K = ctrl._solver.K[0]

    for t in range(r):
        m = read_state()
        feedback = np.dot(K, ctrl.ocp.state.diff(m['x_m'], x) )
        device.joints.set_torques(u + feedback)
        device.send_command_and_wait_end_of_cycle()

        store_measures()

def control_loop(ctrl):
    for t in range(horizon):      
        measures = read_state()
        
        ctrl.create_target(t)
        ctrl.shift_gate()
        ctrl.compute_step(measures['x_m'], ctrl.x0, ctrl.u0)
        
        send_torques(ctrl)  

if __name__ == '__main__':
    ctrl = Controller(10, 50, dt_ocp)
    local_res = Results()
    device = Init_simulation(ctrl.qj0)
    store_measures()

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


    plot_ocp(ctrl, ctrl.results, local_res, 0.001)
