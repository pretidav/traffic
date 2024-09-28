import random 
import numpy as np

class SingleLane():
    def __init__(self, L: int, N: int, bc: str = 'periodic', seed:int = 0, vmax:int = 5):
        self.L = L 
        self.N = N 
        self.allowed_bc = ['periodic','open']
        self.bc = self.validate_bc(bc=bc)
        self.seed = seed 
        self.vmax = vmax
        random.seed(seed)

        self.x,self.v  = self.initialize_xv()
        
    def initialize_xv(self) -> np.array: 
        pos = np.random.choice(a=self.L,size=self.N,replace=False)
        assert(len(pos)==self.N)
        x = np.zeros(shape=self.L,dtype='int32')
        x[pos]=1
        v = np.where(x,np.random.randint(low=0,high=self.vmax,size=self.L),0)
        return np.sort(pos),v

    def validate_bc(self,bc:str) -> str:
        if bc in self.allowed_bc:
            return bc 
        else: 
            raise KeyError(f"boundary condition bc: '{bc}' not in {self.allowed_bc}")
        
    def __repr__(self) -> str:
        return str(self.__dict__)



if __name__=='__main__': 
    lane = SingleLane(L=10,N=5)
    print(lane)
