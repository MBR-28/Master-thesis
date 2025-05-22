import numpy as np

x,y = 1,2

q = np.array([[x,y],
              [3,4]]).T

oi = np.array([0,1])
L=  x*oi@q

#print(L)

#print(np.random.rand(2))


m=1
g=10
koord = np.array([[0,1,2,3]])
print(np.shape(koord)[0])
def Lagrangianxy(koord):
    x,y,dx,dy = np.split(koord,4)
    L= 0.5*m*(dx**2 + dy**2) - m*g*y
    L = L.item()
    return L

print(Lagrangianxy(koord.T))

