from utils.PyBulletSimulator import PyBulletSimulator
from utils.plot_utils import plot_ocp
import numpy as np
from Controller_ipopt import Controller, Results
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

def send_torques():
    u = ctrl.results.ocp_storage['us'][0]
    q = ctrl.results.ocp_storage['qj_des'][0]
    v = ctrl.results.ocp_storage['vj_des'][0]
    for i in range(len(u)):
        qi, vi = interpolate_traj(q[i], v[i])
        for t in range(r):
            device.joints.set_desired_positions(qi[t])
            device.joints.set_desired_velocities(vi[t])
            device.joints.set_position_gains(4)
            device.joints.set_velocity_gains(0.1)
            device.joints.set_torques(u[i])
            device.send_command_and_wait_end_of_cycle()

            store_measures()

def control_loop(ctrl):

    ctrl.create_target(0)

    try:
        guesses = np.load('/tmp/sol_crocoddyl.npy',allow_pickle=True).item()
        ws = {'xs': list(guesses['xs']), 'us': list(guesses['us']), \
            'acs':guesses['acs'] , 'fs':guesses['fs'] }
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

    viz.play(ctrl.results.ocp_storage['xs'][1][:, :19].T, dt_ocp) # SHOW OCP RESULT
    #viz.play(local_res.x_m[:, :19].T, dt_sim) # SHOW PYBULLET SIMULATION
    print("Total cost is: ", ctrl.ocp.opti.value(ctrl.ocp.cost))

    np.save(open('/tmp/sol_ipopt.npy', "wb"),
        {
            "xs": ctrl.results.ocp_storage['xs'][1],
            'acs': ctrl.results.ocp_storage['acs'][0],
            "us": ctrl.results.ocp_storage['us'][0],
            'fs': ctrl.results.ocp_storage['fs'][0] } )

    #plot_ocp(ctrl, ctrl.results, local_res, 0.001)
