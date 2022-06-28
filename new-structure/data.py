class OcpData:
    def __init__(self):
        self.ocp_storage = {'xs': [], 'us': [], 'fw': [], 'qj_des': [], 'vj_des': [], 'residuals' : {'inf_pr': [], 'inf_du': []}}

class ModelData:
    def __init__(self):
        self.x = None
        self.a = None
        self.u = None
        self.f = None
        self.xnext = None
        self.cost = None