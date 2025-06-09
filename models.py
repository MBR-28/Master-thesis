import torch as th
from utilities import nonlin_choice

"""
Simplifying functions for readability
"""
linear = th.nn.Linear
orthogonal_init = th.nn.init.orthogonal_
"""
Basic NN model 
"""


class BasicNN(th.nn.Module):
    def __init__(self, input_size, output_size, nonlinearity , hidden = [20,20,20]):
        super(BasicNN,self).__init__()
        #init input and outpu layer
        self.linear1 = linear(input_size,hidden[0])
        self.linear3 = linear(hidden[-1],output_size,bias=None)

        orthogonal_init(self.linear1.weight)
        orthogonal_init(self.linear3.weight)

        #init hidden layers
        hidden.append(hidden[-1])
        self.layers = th.nn.ModuleList([th.nn.Linear(hidden[i],hidden[i+1]) for i in range(len(hidden)-1)])
        for j in self.layers:
            orthogonal_init(j.weight)

        #print(len(self.layers))

        self.nonlin = nonlin_choice(nonlinearity)

    def forward(self, x):

        y = self.nonlin(self.linear1(x))
        #print(len(self.layers))
        for j in self.layers:
            
            y = self.nonlin(j(y))
            
            

        return self.linear3(y)
            






"""
HNN model based on greydanus Hamiltonian neural Network
"""

class HNNG(th.nn.Module):
    def __init__(self, input_size, output_size, nonlinearity , 
                 hidden = [20,20,20], separate = False,use_cuda=True):
        super(HNNG,self).__init__()
        if True==use_cuda:
            self.M = self.permutation_tensor(input_size).cuda()
            self.eye = th.eye(*self.M.shape).cuda()
            self.base_model = BasicNN(input_size,output_size,nonlinearity,hidden).cuda()
        else:
            self.M = self.permutation_tensor(input_size)
            self.eye = th.eye(*self.M.shape)
            self.base_model = BasicNN(input_size,output_size,nonlinearity,hidden)
        self.separate = separate

    def forward(self, inp):
        # Based on the code associated https://arxiv.org/pdf/1906.01563
        F1, F2 =  self.base_model(inp).split(1,1)

        

        conservative = th.zeros_like(inp)
        solenoidal = th.zeros_like(inp)

        

        dF1 = th.autograd.grad(F1.sum(),inp,create_graph=True)[0]
        conservative = dF1 @ self.eye
        dF2 = th.autograd.grad(F2.sum(),inp,create_graph=True)[0]
        solenoidal = dF2 @ self.M.t()

        if True ==self.separate:
            return [conservative,solenoidal]
        
        return conservative + solenoidal
    
    def permutation_tensor(self,n):
        '''Constructs the Levi-Civita permutation tensor'''
        M = th.ones(n,n) # matrix of ones
        M *= 1 - th.eye(n) # clear diagonals
        M[::2] *= -1 # pattern of signs
        M[:,::2] *= -1

        for i in range(n): # make asymmetric
            for j in range(i+1, n):
                M[i,j] *= -1
        
        return M


"""
HNN model with a lagrangian tau calculation: BAD IDEA, THE GRADIENT CALCULATION METHOD DID NOT WORK
"""

"""
HNN with a NN calculated tau (simmilar to Dissipative HNN) Sosanya: 
NO TIME, STILL NO GOOD WAY TO FIND TAU, SEPARATE TAU CPOORD WOULD ALSO NEED ALMOST FULL CODE REWORK
"""


