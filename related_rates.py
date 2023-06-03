import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
from numba import jit
import time

@jit(nopython=True)
def RK_loop(t, x, f_dxdt, dt, rate_matrix):
    
    '''
    Implements an RK4 method.
    '''
    
    for i in range(0, len(t)-1):#
        
        k1 = f_dxdt(x[i,:], t[i], rate_matrix)
        k2 = f_dxdt(x[i,:] + k1*dt/2, t[i] + dt/2, rate_matrix) 
        k3 = f_dxdt(x[i,:] + k2*dt/2, t[i] + dt/2, rate_matrix) 
        k4 = f_dxdt(x[i,:] + k3*dt, t[i] + dt, rate_matrix) 
        
        x[i+1,:] = x[i,:] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        
    return x
    
@jit(nopython=True)
def f_dxdt(xi,t,rate_matrix):
    '''
    Equation of motion for related rates problem. The off-diagonal elements
    of the rate matrix represent transition rates.
    '''
    
    dxdt = np.dot(rate_matrix.transpose(), xi) - np.sum(rate_matrix,axis=1)*xi
    
    return dxdt

def related_rates_problem(rate_matrix, x0, t):
    '''
    Wrapper for the RK loop that creates all the necessary arrays, since
    you can't create arrays inside of a jit function.
    '''

    N = len(x0)
    
    x = np.zeros([len(t),N])
    
    dt = t[1] - t[0]
    
    x[0,:] = x0

    x = RK_loop(t, x, f_dxdt, dt, rate_matrix)
    
    return x
    
 

if __name__ == '__main__':
    
    time_start = time.time()
    
    t = np.linspace(0,1000,10001)

    N = 15
     
    rate_matrix = np.random.rand(N,N)
    rate_matrix *= (np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1))
    
    x0 = np.zeros(N)
    x0[0] = 1
    
   
    x = related_rates_problem(rate_matrix, x0, t)
    
    print(time.time()-time_start)
    
    plt.semilogx(t,x)