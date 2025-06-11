import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path
Current_Directory = Path(__file__).parent
Par_Dir = Current_Directory.parent
Fig_Save_Dir = Par_Dir /'Fig'


tanh = np.tanh

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

x = np.linspace(-4,4,100)
flat = np.zeros_like(x)

short = np.linspace(-2,2,100)
shortflat = np.zeros_like(short)

y1 = tanh(x)
y2 = sigmoid(x)

plt.plot(x,y1,'b-',label='tanh')
plt.plot(x,y2,'r-',label='sigmoid')
plt.plot(x,flat,'k-')
plt.plot(shortflat,short,'k-')

plt.xlim([-4,4])
plt.ylim([-1.5,1.5])

plt.xlabel('In')
plt.ylabel('Out')

plt.title('tanh and sigmoid')

plt.legend()
plt.grid()

plt.savefig(fname=Fig_Save_Dir/'Tanhsigmoid.png',dpi=300)
plt.show()