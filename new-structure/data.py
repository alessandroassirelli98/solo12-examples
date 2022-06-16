class Data:
    def __init__(self):
        self.xs = []
        self.us = []

        self.tau_ff =  []
        self.tau = []
        self.x_m = []
        self.ocp_storage = {'xs': [], 'us': [], 'fw': [], 'qj_des': [], 'vj_des': [], 'residuals' : {'inf_pr': [], 'inf_du': []}}