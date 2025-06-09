import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
Current_Directory = Path(__file__).parent
Par_Dir = Current_Directory.parent
Fig_Save_Dir = Par_Dir /'Fig'

def f(x):
    #return -np.pow(x,3) +3*np.pow(x,2) -2 
    return np.cos(x)

l = 3
h = 2.5


x = np.linspace(-l,l,100)
y = f(x)

xsample = np.linspace(-l,l,50)

ysample = f(xsample)


ysample += (np.random.rand(np.shape(ysample)[0])-0.5)*2


underfit = np.polynomial.Polynomial.fit(xsample,ysample, deg = 1)
yu = underfit(x)

overfit = np.polynomial.Polynomial.fit(xsample,ysample,deg=21)
yo = overfit(x)

#Inizialize subplot function
fig,ax = plt.subplots(2,2)

#Point cloud
ax[0,0].plot(xsample,ysample,'k.')
ax[0,0].set_xlim(-l,l)
ax[0,0].set_ylim(-h,h)
ax[0,0].grid()
ax[0,0].set_title('Data')
ax[0,0].set_xlabel('In')
ax[0,0].set_ylabel('Out')

#Underfit
ax[0,1].plot(xsample,ysample,'k.')
ax[0,1].set_xlim(-l,l)
ax[0,1].set_ylim(-h,h)
ax[0,1].grid()
ax[0,1].plot(x,yu,'r-')
ax[0,1].set_title('Underfit')
ax[0,1].set_xlabel('In')
ax[0,1].set_ylabel('Out')



#Good fit
ax[1,0].plot(xsample,ysample,'k.')
ax[1,0].set_xlim(-l,l)
ax[1,0].set_ylim(-h,h)
ax[1,0].grid()
ax[1,0].plot(x,y,'r-')
ax[1,0].set_title('Good fit')
ax[1,0].set_xlabel('In')
ax[1,0].set_ylabel('Out')


#Overfit
ax[1,1].plot(xsample,ysample,'k.')
ax[1,1].set_xlim(-l,l)
ax[1,1].set_ylim(-h,h)
ax[1,1].grid()
ax[1,1].plot(x,yo,'r-')
ax[1,1].set_title('Overfit')
ax[1,1].set_xlabel('In')
ax[1,1].set_ylabel('Out')



plt.subplots_adjust(wspace=0.5,hspace=0.5)

plt.savefig(fname=Fig_Save_Dir/'Fitting.png')
plt.show()
