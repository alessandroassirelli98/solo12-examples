#from ocp import SimpleManipulationProblem
from controller import Controller
from problem import ProblemData
from utils.PyBulletSimulator import PyBulletSimulator
from pinocchio.visualize import GepettoVisualizer
import numpy as np

pd = ProblemData()

horizon = 1
dt_ocp = pd.dt
dt_sim = 0.001
r = int(dt_ocp/dt_sim) 

#device = Init_simulation(pd.x0[: pd.nq])
pd.create_target(0)
ctrl = Controller(pd, dt_sim, r, 'crocoddyl')

def control_loop():
    for t in range(horizon):      
        m = ctrl.read_state()
        
        pd.create_target(t)
        if t != 0:
            pd.shift_gate()
        ctrl.compute_step(pd.x0)

control_loop()

try:
    viz = GepettoVisualizer(pd.model, pd.robot.collision_model, pd.robot.visual_model)
    viz.initViewer()
    viz.loadViewerModel()
    gv = viz.viewer.gui
except:
    print("No viewer"  )

#viz.play(solver.results.ocp_storage['xs'][0][:, :19].T, pd.dt) # SHOW OCP RESULT
