from ocp import SimpleManipulationProblem
from problem import ProblemData
from utils.PyBulletSimulator import PyBulletSimulator
from pinocchio.visualize import GepettoVisualizer
import numpy as np

pd = ProblemData()

horizon = 10
dt_ocp = pd.dt
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
    u = solver.results.ocp_storage['us'][-1][0]
    x = solver.results.ocp_storage['xs'][-1][0]
    K = solver._solver.K[0]
    for t in range(r):
        m = read_state()
        feedback = np.dot(K, solver.state.diff(m['x_m'], x))
        device.joints.set_torques(u + feedback)
        device.send_command_and_wait_end_of_cycle()

        store_measures()

def get_mpc_sol():
    q_mpc = []
    [q_mpc.append(ctrl.results.ocp_storage['xs'][i][1, :19]) for i in range(horizon)]
    
    return np.array(q_mpc)

def control_loop(ctrl):
    for t in range(horizon):      
        m = read_state()
        
        pd.create_target(t)
        ctrl.shift_gate()
        solver.solve(m['x_m'], pd.x0, pd.u0)
        
        send_torques()   

device = Init_simulation(pd.x0[: pd.nq])
solver = SimpleManipulationProblem("crocoddyl", pd)


try:
    viz = GepettoVisualizer(pd.model, pd.robot.collision_model, pd.robot.visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    gv = viz.viewer.gui
except:
    print("No viewer"  )

viz.play(solver.results.ocp_storage['xs'][0][:, :19].T, pd.dt) # SHOW OCP RESULT
