import numpy as np
import matplotlib.pyplot as plt

class myGrid():
    """ a finite differnece grid"""

    # constructor method for myGrid class
    def __init__(self, nx, ng=1,xmin=0, xmax=1):
        #create a grid with nx nodes and ng ghost nodes on each side
        #domain ranges from [xmin, xmax]
        #nx = total number of physical grid points
        #ng = no. of ghost nodes on each side

        self.xmin = xmin                 #initialization as well as value assignment
        self.xmax = xmax
        self.ng = ng
        self.nx = nx

        #making indices for finding physical grid

        self.ibegin = ng
        self.iend = nx+ng-1

        #physical grid points
        self.dx = (xmax - xmin)/ (nx-1)
        self.xcoords = xmin + (np.arange(nx+2*ng)-ng)*self.dx

        #array for storing solution
        self.phi = np.zeros((self.nx+2*self.ng),dtype = np.float64)
        self.phi_init = np.zeros((self.nx+2*self.ng),dtype = np.float64)

    def array_memory_allocator(self):
        #allocate memory for an arbitrary array having same dimension as the grid
        return np.zeros((self.nx+2*self.ng),dtype = np.float64)

        
    def apply_BCs(self):
        #updating the single ghost node with periodic BCs

        self.phi[self.ibegin -1] = self.phi[self.iend -1]   # (-1) = (N-2)  left ghost node 
        self.phi[self.iend+1] = self.phi[self.ibegin +1]   # (nx+ng) = (ng+1) or N+1 = 1  right ghost node

def diffusion(nx, alpha, Fo, num_periods = 1.0, init_cond = None):
    #solve 1-D linear advection using FOU
    #Inputs: pass a function f(g), where g is grid object, that sets up the initial condition

    #creating grid object using myGrid Class
    g= myGrid(nx)
        
    #computing the time step, dt

    dt = (Fo*(g.dx)**2)/alpha
    t_start = 0.0
    t_max = 0.1        #time = distance/speed
    t_current = t_start

    #initialize the data
    init_cond(g)
    g.phi_init[:] = g.phi[:]  #copy the initial condition to uinit

    #Time integration
    phi_new = g.array_memory_allocator()

    while t_current<t_max:
        if t_current +dt >t_max:
            dt = t_max - t_current
            Fo = alpha*dt/(g.dx)**2

        #apply BCs
        g.apply_BCs()

        #FTCS Update
        for i in range(g.ibegin,g.iend+1):
            phi_new[i] = g.phi[i] + alpha*dt*(g.phi[i+1]-2*g.phi[i]+g.phi[i-1])/(g.dx)**2
        
        g.phi_exact = g.array_memory_allocator()
        g.phi_exact = 0.6*np.exp(-4*((np.pi)**2)*alpha*t_current)*np.sin(2*np.pi*g.xcoords) + 0.3*np.exp(-16*((np.pi)**2)*alpha*t_current)*np.sin(4*np.pi*g.xcoords)+0.1*np.exp(-64*((np.pi)**2)*alpha*t_current)*np.sin(8*np.pi*g.xcoords)

        g.phi[:] = phi_new[:]

        t_current += dt

    return g


#function to initialize the state of 
def sine_wave(g):
    g.phi[:] = (0.6*np.sin(2*np.pi*g.xcoords) + 0.3*np.sin(4*np.pi*g.xcoords)+0.1*np.sin(8*np.pi*g.xcoords))/4

def error(g):
    g.error_sum = 0 
    N = g.nx+2*g.ng-1
    for i in range(N):
        g.error_val = (g.phi[i] - g.phi_exact[i])**2
        g.error_sum += g.error_val

    g.norm2 = np.sqrt(((g.error_sum)**2)/N)
    print(g.norm2)


    return 0

    
def plot(g):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(g.xcoords[g.ibegin:g.iend+1], g.phi_init[g.ibegin:g.iend+1], label = 'Init Cond')
    ax.plot(g.xcoords[g.ibegin:g.iend+1], g.phi[g.ibegin:g.iend+1], label = 'Simulation')

    ax.plot(g.xcoords[g.ibegin:g.iend+1], g.phi_exact[g.ibegin:g.iend+1], label = "Exact Solution")

    ax.legend()
    plt.show()

nx = 512

g = diffusion(nx,alpha=5e-5,Fo = 0.7, num_periods=1, init_cond=sine_wave)
error(g)
plot(g)



    