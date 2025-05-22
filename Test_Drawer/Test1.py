import random as ran
import autograd.numpy as np

n = 100
theta = []
for i in range(0,100,1):
    t = (ran.randint(0,n)/n)*np.pi
    theta.append(t)
print(theta)