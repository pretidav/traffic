import random 
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm

class SingleLane():
    def __init__(self, L: int, N: int, p:float = 0.3, bc: str = 'periodic', seed:int = 0, vmax:int = 5):
        self.L = L 
        self.N = N 
        self.p = p
        self.allowed_bc = ['periodic','open']
        self.bc = self.validate_bc(bc=bc)
        self.seed = seed 
        self.vmax = vmax
        random.seed(seed)
        self.x,self.v  = self.initialize_xv()
        self.d = self.distances()
        self.stringx = self.plot(xORv='x')
        self.stringv = self.plot(xORv='v')
        
    def initialize_xv(self) -> np.array: 
        x = np.random.choice(a=self.L,size=self.N,replace=False)
        assert(len(x)==self.N)
        v = np.random.randint(low=0,high=self.vmax,size=self.N)
        return np.sort(x),v

    def validate_bc(self,bc:str) -> str:
        if bc in self.allowed_bc:
            return bc 
        else: 
            raise KeyError(f"boundary condition bc: '{bc}' not in {self.allowed_bc}")
        
    def __repr__(self) -> str:
        return str(self.__dict__)
    
    def pos2idx(self,pos: int) -> int:
        return np.where(self.x == pos)[0]
    
    def plot(self, xORv:'str' = 'x') -> str:
        pos = '|'
        for i in range(self.L):
            if i in self.x and xORv=='x': 
                pos+='x'
            elif i in self.x and xORv=='v':
                j = self.pos2idx(i)
                pos+=str(self.v[j][0])
            else: 
                pos+='-'
        pos+='|'
        return pos 
    
    def update_plot(self): 
        self.stringx = self.plot(xORv='x')
        self.stringv = self.plot(xORv='v')
    
    def distances(self) -> np.array: 
        if self.bc=='periodic':
            d = []
            d = np.roll(self.x, shift=-1, axis=0)[:-1] - self.x[:-1]
            d = np.append(d, self.x[0] + (self.L-self.x[-1]))
            return np.abs(d)
        else:
            print('TBD') 
            exit(1)
        
    def acceleration(self): 
        for pos in list(self.x):
            idx = self.pos2idx(pos)
            if self.v[idx] < self.vmax and self.d[idx] > self.v[idx]+1:
                self.v[idx]+=1

    def slowing_down(self): 
        for pos in list(self.x):
            idx = self.pos2idx(pos)
            if self.d[idx] <= self.v[idx]:
                self.v[idx]=self.d[idx]-1
    
    def randomization(self): 
        rnd_slow = np.random.rand(self.N)<self.p
        self.v = np.where(rnd_slow,self.v-1,self.v)
        self.v=np.where(self.v>0,self.v,0)
        
    def car_motion(self):      
        tmp = np.zeros_like(self.x)
        for idx in range(self.N):
            tmp[idx]=(self.x[idx]+self.v[idx]) % (self.L)  
        self.x = tmp
        self.sort()
        self.d = self.distances()
    
    def sort(self): 
        idx_sort=np.argsort(self.x)
        self.x=self.x[idx_sort]
        self.v=self.v[idx_sort]
        self.d=self.d[idx_sort]
        
    def update(self): 
        self.acceleration()
        self.slowing_down()
        self.randomization()
        self.car_motion()
        self.update_plot()
                

                
if __name__=='__main__': 
    N = 500 # tot number of cars
    L = 1000 # road length
    T =  50 # tot duration evolution
    p = 0.3 # slowing down prob
    vmax = 100 # max speed allowed
    
    v = []
    dv = []
    for n in tqdm(range(1,N)):        
        lane = SingleLane(L=L,N=n,p=p,vmax=vmax)
        v_mean = []
        for n in range(T):
            lane.update()
            v_mean.append(np.mean(lane.v))
        v.append(np.mean(v_mean))
        dv.append(np.std(v_mean)/np.sqrt(T))
    
    plt.errorbar(x=np.array(list(range(1,N)))/L, y=v, yerr=dv, marker='.', linestyle='')
    plt.xlabel('traffic density N/L')
    plt.ylabel('<v>')
    plt.savefig('flow_N{}_L{}_T{}_p{}_vmax{}.png'.format(N,L,T,p,vmax))