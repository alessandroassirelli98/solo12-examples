import os
import sys
import time

import example_robot_data
import numpy as np
from utils import loader, PyBulletSimulator

import crocoddyl
import pinocchio
from utils.quadruped_manipulation import SimpleManipulationProblem, plotSolution

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

dt_sim = 0.001

# Loading the Solo12 model
solo = loader.Solo12()
x0 = np.concatenate([solo.q0, solo.v0])
effort_limit = np.ones(solo.robot.nv - 6) *3   
solo.model.effortLimit = effort_limit

""" solo = example_robot_data.load("anymal")

q0 = solo.model.referenceConfigurations['standing'].copy()
v0 = pinocchio.utils.zero(solo.model.nv)
x0 = np.concatenate([q0, v0]) """

def Init_simulation(q_init):
    device = PyBulletSimulator()
    device.Init(q_init, 0, True, True, dt_sim)
    return device


# Setting up the 3d walking problem
lfFoot, rfFoot, lhFoot, rhFoot = 'FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT'
#lfFoot, rfFoot, lhFoot, rhFoot = 'LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT'
gait = SimpleManipulationProblem(solo.model, lfFoot, rfFoot, lhFoot, rhFoot)

walking_gait = {'stepLength': 0.15, 'stepHeight': 0.15, 'timeStep': 1e-2, 'stepKnots': 100, 'supportKnots': 10}

# Setting up the control-limited DDP solver
solver = crocoddyl.SolverBoxDDP(
    gait.createMovingFootProblem(x0, walking_gait['stepLength'], walking_gait['stepHeight'], walking_gait['timeStep'],
                              walking_gait['stepKnots'], walking_gait['supportKnots']))

# Add the callback functions
print('*** SOLVE ***')
cameraTF = [2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]
if WITHDISPLAY and WITHPLOT:
    display = crocoddyl.GepettoDisplay(solo.robot, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(solo.robot, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])


# Solve the DDP problem
xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.solve(xs, us, 100, False, 0.1)


# Plotting the entire motion
if WITHPLOT:
    # Plot control vs limits
    plotSolution(solver, bounds=True, figIndex=1, show=False)

    # Plot convergence
    log = solver.getCallbacks()[0]
    crocoddyl.plotConvergence(log.costs,
                              log.u_regs,
                              log.x_regs,
                              log.grads,
                              log.stops,
                              log.steps,
                              figIndex=3,
                              show=True)

# Display the entire motion
WITHDISPLAY = True
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(solo.robot, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    while True:
        display.displayFromSolver(solver)
        time.sleep(2.0)