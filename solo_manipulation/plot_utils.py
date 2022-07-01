import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from kinematics_utils import get_translation, get_translation_array
from Controller import Controller

plt.style.use("seaborn")

def plot_ocp(ctrl, ocp_results, local_results, dt_simu):
    r = int(ctrl.dt/dt_simu)

    t15 = np.linspace(0, (ctrl.ocp.T)*ctrl.dt, ctrl.ocp.T+1)
    t1 = np.linspace(0, (ctrl.ocp.T)*ctrl.dt, ctrl.ocp.T*r + 1)

### ------------------------------------------------------------------------- ###
    # FORCES IN WORLD FRAME 
    forces = ocp_results.ocp_storage['fw'][0]
    legend = ['F_x', 'F_y', 'F_z']
    plt.figure(figsize=(12, 6), dpi = 90)
    for i, foot in enumerate(forces):
        plt.subplot(2,2,i+1)
        plt.title('OCP Forces on ' + str(i))
        [plt.plot(t15[:-1], forces[foot][:, jj]) for jj in range(3) ]
        plt.ylabel('Force [N]')
        plt.xlabel('t[s]')
        plt.legend(legend)
    plt.draw()

### ------------------------------------------------------------------------- ###
    # JOINT VELOCITIES
    x = ocp_results.ocp_storage['xs'][1]
    legend = ['Hip', 'Shoulder', 'Knee']
    plt.figure(figsize=(12, 6), dpi = 90)
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.title('Joint velocity of ' + str(i))
        [plt.plot(x[:, 19 + (3*i+jj)]*180/np.pi ) for jj in range(3) ]
        plt.ylabel('Velocity [Deg/s]')
        plt.xlabel('t[s]')
        plt.legend(legend)
    plt.draw()

### ------------------------------------------------------------------------- ###
    # JOINT TORQUES
    u = ocp_results.ocp_storage['us'][0]
    legend = ['Hip', 'Shoulder', 'Knee']
    plt.figure(figsize=(12, 6), dpi = 90)
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.title('Joint torques of ' + str(i))
        [plt.plot(u[:, (3*i+jj)]) for jj in range(3) ]
        plt.ylabel('Velocity [Deg/s]')
        plt.xlabel('t[s]')
        plt.legend(legend)
    plt.draw()

### ------------------------------------------------------------------------- ###
    # BASE POSITION

    base_log_ocp = ctrl.ocp.get_base_log(ocp_results.ocp_storage['xs'][1])
    base_log_m = ctrl.ocp.get_base_log(local_results.x_m)

    legend = ['x', 'y', 'z']
    plt.figure(figsize=(12, 6), dpi = 90)
    for i in range(3):
        for foot in ctrl.ocp.terminalModel.freeIds:
            plt.subplot(3,1,i+1)
            plt.title('Base position on ' + legend[i])
            plt.plot(t15, base_log_ocp[:, i])
            plt.plot(t1, base_log_m[:, i])
            plt.legend(['OCP', 'BULLET'])
    plt.draw()

### ------------------------------------------------------------------------- ###
    # FEET POSITIONS 
    feet_log_ocp = ctrl.ocp.get_feet_position(ocp_results.ocp_storage['xs'][1])
    feet_log_m = ctrl.ocp.get_feet_position(local_results.x_m)

    legend = ['x', 'y', 'z']
    plt.figure(figsize=(12, 6), dpi = 90)
    for i in range(3):
        for foot in ctrl.ocp.terminalModel.freeIds:
            plt.subplot(3,1,i+1)
            plt.title('Foot position on ' + legend[i])
            plt.plot(t15, feet_log_ocp[foot][:, i])
            plt.plot(t1, feet_log_m[foot][:, i])
            plt.legend(['OCP', 'BULLET'])
    plt.draw()



    plt.show()


def plot_mpc(ctrl:Controller):

    horizon = len(ctrl.results.ocp_storage['xs'])
    t15 = np.linspace(0, horizon*ctrl.pd.dt, horizon+1)
    t1 = np.linspace(0, (horizon)*ctrl.pd.dt, (horizon)*ctrl.pd.r1+1)
    t_mpc = np.linspace(0, (horizon)*ctrl.pd.dt, horizon+1)

    all_ocp_xs = [np.array([ctrl.pd.x0] * len(ctrl.results.ocp_storage['xs'][0]))]
    [all_ocp_xs.append(x) for x in ctrl.results.ocp_storage['xs'] ]
    all_ocp_xs = np.array(all_ocp_xs)

    x_mpc = [x[1, :] for x in all_ocp_xs]
    x_mpc = np.array(x_mpc)

    feet_p_log_mpc = {id: get_translation_array(ctrl.pd, x_mpc, id)[0] for id in ctrl.pd.allContactIds}
    feet_p_log_m = {id: get_translation_array(ctrl.pd, ctrl.results.x_m, id)[0] for id in ctrl.pd.allContactIds}
    feet_v_log_mpc = {id: get_translation_array(ctrl.pd, x_mpc, id)[1] for id in ctrl.pd.allContactIds}

    all_ocp_feet_p_log = {idx: [get_translation_array(ctrl.pd, x, idx)[0] for x in all_ocp_xs] for idx in ctrl.pd.allContactIds}
    for foot in all_ocp_feet_p_log: all_ocp_feet_p_log[foot] = np.array(all_ocp_feet_p_log[foot])
    
    legend = ['x', 'y', 'z']
    plt.figure(figsize=(12, 6), dpi = 90)
    for i in range(3):
        for foot in [18]: # Plot only the foot which is free at terminal node
            plt.subplot(3,1,i+1)
            plt.title('Foot position on ' + legend[i])
            plt.plot(t15, feet_p_log_mpc[foot][:, i])
            plt.plot(t1, feet_p_log_m[foot][:, i])
            plt.legend(['OCP', 'BULLET'])
    plt.draw()
    

    
    legend = ['x', 'y', 'z']
    plt.figure(figsize=(12, 18), dpi = 90)
    for p in range(3):
        plt.subplot(3,1, p+1)
        plt.title('Free foot on ' + legend[p])
        for i in range(horizon):
            t = np.linspace(i*ctrl.pd.dt, (ctrl.pd.T+ i)*ctrl.pd.dt, ctrl.ocp.pd.T+1)
            y = all_ocp_feet_p_log[18][i+1][:,p]
            for j in range(len(y) - 1):
                plt.plot(t[j:j+2], y[j:j+2], color='royalblue', linewidth = 3, marker='o' ,alpha=max([1 - j/len(y), 0]))
        plt.plot(t_mpc, feet_p_log_mpc[18][:, p], linewidth=0.8, color = 'tomato', marker='o')
        plt.plot(t1, feet_p_log_m[18][:, p], linewidth=2, color = 'lightgreen')
        #plt.ylim([0.18, 0.25])


    plt.figure(figsize=(12, 24), dpi = 90)
    for q in range(12):
        plt.subplot(6,2, q+1)
        plt.title('q ' + str(q+7))
        for i in range(horizon):
            t = np.linspace(i*ctrl.pd.dt, (ctrl.pd.T+ i)*ctrl.pd.dt, ctrl.ocp.pd.T+1)
            y = all_ocp_xs[i+1][:, 7+q]
            for j in range(len(y) - 1):
                plt.plot(t[j:j+2], y[j:j+2], color='royalblue', linewidth = 3, marker='o' ,alpha=max([1 - j/len(y), 0]))
        plt.plot(t_mpc, x_mpc[:, 7 + q], linewidth=0.8, color = 'tomato', marker='o')
        plt.plot(t1, ctrl.results.x_m[:, 7+ q], linewidth=2, color = 'lightgreen')
        #plt.ylim([0.18, 0.25])
    plt.draw()
    plt.show()

    


    """u_mpc = local_results.tau_ff
    u_mpc = np.array(u_mpc)
    all_u_log = np.array(ocp_results.ocp_storage['us'])
    u_m = np.array(local_results.tau)

    legend = ['x', 'y', 'z']
    plt.figure(figsize=(12, 6), dpi = 90)
    for i in range(3):
        for foot in ctrl.ocp.terminalModel.freeIds:
            plt.subplot(3,1,i+1)
            plt.title('Foot position on ' + legend[i])
            plt.plot(t15, feet_log_mpc[foot][:, i])
            plt.plot(t1, feet_log_m[foot][:, i])
            plt.legend(['OCP', 'BULLET'])
    plt.draw()

    legend = ['x', 'y', 'z']
    plt.figure(figsize=(12, 18), dpi = 90)
    for p in range(3):
        plt.subplot(3,1, p+1)
        plt.title('Free foot on ' + legend[p])
        for i in range(horizon):
            t = np.linspace(i*ctrl.dt, (ctrl.ocp.T+ i)*ctrl.dt, ctrl.ocp.T+1)
            y = all_ocp_feet_log[i+1][:,p]
            for j in range(len(y) - 1):
                plt.plot(t[j:j+2], y[j:j+2], color='royalblue', linewidth = 3, marker='o' ,alpha=max([1 - j/len(y), 0]))
        plt.plot(t_mpc, feet_log_mpc[18][:, p], linewidth=0.8, color = 'tomato', marker='o')
        plt.plot(t1, feet_log_m[18][:, p], linewidth=2, color = 'lightgreen')
    plt.draw()


    plt.figure(figsize=(12, 36), dpi = 90)
    for p in range(12):
        plt.subplot(12,1, p+1)
        plt.title('u ' + str(p))
        for i in range(horizon):
            t = np.linspace(i*ctrl.dt, (ctrl.ocp.T+ i)*ctrl.dt, ctrl.ocp.T+1)
            y = all_u_log[i][:,p]
            for j in range(len(y) - 1):
                plt.plot(t[j:j+2], y[j:j+2], color='royalblue', linewidth = 3, marker='o' ,alpha=max([1 - j/len(y), 0]))

            plt.plot(t_mpc[:-1], u_mpc[:, p], linewidth=0.8, color = 'tomato', marker='o')
            plt.plot(t1, u_m[:, p], linewidth=2, color = 'lightgreen')
    plt.draw()


    plt.show() """