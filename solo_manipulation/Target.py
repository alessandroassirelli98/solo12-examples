import numpy as np
from ProblemData import ProblemData

class Target:
    def __init__(self, pd:ProblemData):
        self.pd = pd
        self.dt = pd.dt

        self.gait = [] \
            + [ [ 1,1,1,1 ] ] * pd.init_steps \
            + [ [ 1,0,1,1 ] ] * pd.target_steps

        self.T = pd.T
        self.contactSequence = [ self.patternToId(p) for p in self.gait]

        self.target = {pd.rfFootId: []}
        self.FR_foot0 = np.array([0.1946, -0.16891, 0.017])
        self.A = np.array([0, 0.05, 0.05])
        self.offset = np.array([0.05, 0., 0.06])
        self.freq = np.array([0, 2.5, 2.5])
        self.phase = np.array([0, 0, np.pi/2])

    def patternToId(self, gait):
            return tuple(self.pd.allContactIds[i] for i,c in enumerate(gait) if c==1 )
        
    def shift_gait(self):
        self.gait.pop(0)
        self.gait += [self.gait[-1]]
        self.contactSequence = [ self.patternToId(p) for p in self.gait]    

    def update(self, t):
        target = []
        for n in range(self.T+1):
            target += [self.FR_foot0 + self.offset + self.A *
                       np.sin(2*np.pi*self.freq * (n+t) * self.dt + self.phase)]
        self.target[self.pd.rfFootId] = np.array(target)
    
    def evaluate_in_t(self, t):
        return {e: self.target[e][t] for e in self.target}
