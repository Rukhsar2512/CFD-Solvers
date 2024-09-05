#Solving Diffusion equation using implicit scheme.
""" Graphite rod of 1m with dirichlet condition at leftmost end and neumann condition at rightmost end. 
Transient Solver 
Diffusion Equation - dT/dt = alpha* d^2(T)/dx^2
Initial condtion - T = 0 C
Boundary Condition:
        T(x=0) = 100
        dT/dx|(x=1) = 0

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline
from scipy import linalg


#Simulation Parameters
L = 1.0  #length of rod
alpha = 1.22e-3  #thermal diffusivity of the graphite rod
q = 0.0   #temp gradient on the right boundary

nx = 101        #total no of grid points including the boundary nodes
dx = L/(nx-1)   #uniform grid spacing

#Grid Generation
x,delta_x = np.linspace(0.0, L, num = nx,retstep=True)

#Initial Condition 
To = np.zeros(nx)
To[0] = 100.0

#Time Integration 
#(Solving a system of linear equation)

def assemble_coeff_matrix(N, Fo):   
    # N= matrix size, Fo = Fourier number
    # Here we'll be assembling the A matrix for AX = B
    # A is basically a tridiagonal matrix
    D= np.diag((2. + 1./Fo)*np.ones(N))
    D[-1,-1] = 1. + 1./Fo     #changing the last element as per the Neumann BC

    #setup the upper and lower diagonal
    U = np.diag(-1.*np.ones(N-1),k=1)   #k = offset 
    L = np.diag(-1.*np.ones(N-1), k=-1)

    #combine U, D, L
    A = D+L+U

    return A

def rhs(T, Fo, qdx):
    b = T[1:-1]/Fo
    # Set Dirichlet boundary condition
    b[0] += T[0]  # T[0] is the left boundary node
    # Set the Neumann boundary condition
    b[-1] += qdx
    return b

#BTCS Algorithm

def backward_euler(To, num_time_steps, dt, dx, alpha, q):
    Fo = alpha * dt / dx**2
    A=assemble_coeff_matrix(len(x)-2, Fo)  #create the coeff matrix for the system of equations

    T = To.copy
    for step in range(num_time_steps):
        b=rhs(T,Fo,q*dx)
        T[-1:-1] = linalg.solve(A,b)
        T[-1] = T[-2] + q*dx      #Applying BC

    return T

Fo= 0.5
dt = Fo*dx**2/alpha
nt=1000
T = backward_euler(To, nt, dt, dx, alpha, q)
