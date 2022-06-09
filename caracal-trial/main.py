import crocoddyl
from crocoddyl.utils.quadruped import plotSolution
import caracal
import example_robot_data
import pinocchio
import numpy as np
import sys
import gaits

# Problem configuration
WITHDISPLAY = 'display' in sys.argv or False
WITHPLOT = 'plot' in sys.argv or False
STEP_LENGTH = 0.2
N_GAITS = 2

# ANYmal robot model
robot = example_robot_data.load("anymal")
model = robot.model
model.effortLimit *= 0.4
model.velocityLimit *= 0.5

# Initial state and its contact placement
q0 = model.referenceConfigurations['standing'].copy()
v0 = pinocchio.utils.zero(model.nv)
x0 = np.concatenate([q0, v0])
data = model.createData()
pinocchio.forwardKinematics(model, data, q0, v0)
pinocchio.updateFramePlacements(model, data)

# Define the gait
gait_generator = gaits.QuadrupedalGaitGenerator(model)
cs = [None] * N_GAITS
for c in range(N_GAITS):
    cs[c] = dict()
    for name in ["LH_FOOT", "LF_FOOT", "RH_FOOT", "RF_FOOT"]:
        oMf = data.oMf[model.getFrameId(name)]
        cs[c][name] = pinocchio.SE3(oMf.rotation, oMf.translation + np.array([c * STEP_LENGTH, 0., 0.]))

for c in range(N_GAITS - 1):
    contacts, stepHeight = [cs[c], cs[c + 1]], 0.25
    if c == 0:
        gait = gait_generator.moveFoot(contacts, 40, 20, 0, 0, stepHeight, True, False)
    else:
        gait += gait_generator.moveFoot(contacts, 40, 20, 0, 0, stepHeight, False, False)

# Create the MPC application
params = caracal.CaracalParams()
params.solverVerbose = True
params.withForceReg = True
params.withImpulseReg = True
mpc = caracal.Caracal(q0, model, gait, params=params)
mpc.start(q0, maxiter=200)

# Display the solution and plot results
import time
WITHDISPLAY=True
WITHPLOT=False
if WITHPLOT:
    caracal.plotContactPhaseDiagram(gait, False, show=True)
    plotSolution(mpc._solver, figIndex=2, show=True)
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(robot, frameNames=gait.contactNames)
    while True:
        display.displayFromSolver(mpc._solver)
        time.sleep(2.0)