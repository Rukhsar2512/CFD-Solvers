"""Advection- Diffusion eq - 
du/dt + cdu/dx = vd^2u/dx^2
c = wave speed
v = diffusion coefficient
Descritization using FTCS method
"""

import numpy as np
import matplotlib.pyplot as plt

#simulation parameters

n= 101
L = 1.0
x = np.linspace(0,L,n)
dx = L/(n-1)

#Initial BC
"""  """
w = 2.0*np.pi          #angular frequency 
v = 0.05               #diffusion coeff
c = 1.0                #wave speed       
u0 = lambda x : np.sin(w*x)

uexact = lambda x,t: u0(x-c*t)*np.exp(-v*w*w*t)

plt.plot(x, u0(x),'o')
plt.grid()
plt.show()