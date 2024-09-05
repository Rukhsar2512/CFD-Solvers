import numpy as np
import matplotlib.pyplot as plt

def func(nx):
    xmin = 0
    xmax = 1
    #nx = 256
    ng = 2

    ibegin = ng  #2
    iend = nx+ ng-1  #257

    dx = (xmax - xmin)/(nx-1)
    xcoords = xmin + (np.arange(nx+2*ng)-ng)*dx
   

    phi = np.zeros(nx+2*ng, dtype = np.float64)
    phi_init = np.zeros(nx+2*ng, dtype = np.float64)

    def apply_BCs(phi,ibegin,iend):

        phi[ibegin-1] = phi[iend-1] #1 = 256
        phi[iend+1] = phi[ibegin+1]  #258 = 3

    def solver(Fo, alpha, dx, init_cond = None):
        dt =  (Fo*dx**2)/(alpha)
        t_start = 0.0
        t_max = 0.1

        t_current = t_start

        init_cond()

        phi_init[:] = phi[:]
        

        phi_new = np.zeros(nx+2*ng, dtype = np.float64)

        while t_current<t_max:
            if t_current + dt >t_max:
                dt = t_max - t_current
                Fo = (dt*alpha)/(dx**2)
            
            apply_BCs(phi_new, ibegin, iend)

            for i in range(ibegin,iend+1):
                phi_new[i] = phi[i] + (alpha*dt/dx**2)*(phi[i+1]-2*phi[i]+phi[i-1])
            
            phi[:] = phi_new[:]

            t_current += dt

            global phi_exact
            phi_exact =  np.zeros(nx+2*ng, dtype = np.float64)
            phi_exact = 0.6*np.exp(-4*((np.pi)**2)*alpha*t_current)*np.sin(2*np.pi*xcoords) + 0.3*np.exp(-16*((np.pi)**2)*alpha*t_current)*np.sin(4*np.pi*xcoords)+0.1*np.exp(-64*((np.pi)**2)*alpha*t_current)*np.sin(8*np.pi*xcoords)

        error_sum = 0 
        N = nx+2*ng-1
        for i in range(N):
            error_val = (phi[i] - phi_exact[i])**2
            error_sum += error_val

        norm2 = np.sqrt(((error_sum)**2)/N)
       
        return norm2
    
    def sine_wave():
        phi[:] = 0.6*np.sin(2*np.pi*xcoords) + 0.3*np.sin(4*np.pi*xcoords)+0.1*np.sin(8*np.pi*xcoords)


    solver(0.1,5e-05,dx,init_cond=sine_wave)
    val = solver(0.7,5e-05,dx,init_cond=sine_wave)
    
    fig = plt.figure()
    ax= fig.add_subplot(111)
    ax.plot(xcoords[ibegin:iend+1],phi_init[ibegin:iend+1], label = 'Init Cond')
    ax.plot(xcoords[ibegin:iend+1],phi[ibegin:iend+1], label="Simulated")
    ax.plot(xcoords[ibegin:iend+1],phi_exact[ibegin:iend+1], label="Exact")
    ax.legend()

    plt.show()

    return val


for i in (32,64,128,256,512,1024):
    val = func(i)
    print("For Nx = ",i," error is =",val)