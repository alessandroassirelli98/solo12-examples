import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")

guesses = np.load('/tmp/sol_ipopt.npy',allow_pickle=True).item()
xs_ipopt = guesses['xs']
us_ipopt = guesses['us']

guesses = np.load('/tmp/sol_crocoddyl.npy',allow_pickle=True).item()
xs_croco = guesses['xs']
us_croco = guesses['us']

for i in range(30):
    plt.plot(xs_ipopt[:, i] - xs_croco[:, i])
    plt.show()