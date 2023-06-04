import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
import time
from numba import njit
from numba.typed import List

@njit
def RK_loop(t, x, f_dxdt, dt, args):

    '''
    Implements an RK4 method.
    '''

    for i in range(0, len(t)-1):#

        k1 = f_dxdt(x[i,:], t[i], args)
        k2 = f_dxdt(x[i,:] + k1*dt/2, t[i] + dt/2, args) 
        k3 = f_dxdt(x[i,:] + k2*dt/2, t[i] + dt/2, args) 
        k4 = f_dxdt(x[i,:] + k3*dt, t[i] + dt, args) 
        
        x[i+1,:] = x[i,:] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    return x

@njit
def delta_phi(t, args):
    
    gamma = args[0]
    w_0 = args[1]
    w_J = args[2]
    phi_DC = args[3]
    w_d = args[4]
    phi_max = args[5]
    w_in = args[6]
    phase_in = args[7]
    amp_in = args[8]
    
    phi_ext = phi_max*np.cos(w_d*t)
    
    return phi_DC + phi_ext

@njit 
def phi_in_func(t, args):

    phi_DC = args[3]
    w_d = args[4]
    phi_max = args[5]
    w_in = args[6]
    phase_in = args[7]
    amp_in = args[8]
    
    phi_in = amp_in*np.cos(w_in*t + phase_in)
    
    dphi_indt = -amp_in*w_in*np.sin(w_in*t + phase_in)
    
    return phi_in, dphi_indt
    

@njit
def f_dxdt(xi, t, args):
    
    alpha = args[0]
    beta = args[1]
    gamma = args[2]
    delta = args[3]
    omega = args[4]

    x1 = xi[0]
    x2 = xi[1]

    dx1dt = x2
    dx2dt = -delta*x2 - alpha*x1 - beta*x1**3 + gamma*np.cos(omega*t)
    
    dxdt = np.array([dx1dt, dx2dt])

    return dxdt

def duffing_osc_i(args):
    '''
    Wrapper for the RK loop that creates all the necessary arrays, since
    you can't create arrays inside of a jit function.
    '''
    alpha = args[0]
    beta = args[1]
    gamma = args[2]
    delta = args[3]
    omega = args[4]

    t = np.linspace(0,1000,200000)
    
    N = 2
    
    x0 = np.zeros(N)
    
    x = np.zeros([len(t),N])
    
    dt = t[1] - t[0]
    
    x[0,:] = x0

    result = RK_loop(t, x, f_dxdt, dt, args)
    
    x = result[:,0]
    dxdt = result[:,1]

    traj = x + 1j*dxdt
    
    return t, traj, x, dxdt

def duffing_osc(args_list):

    args = List()
    [args.append(arg) for arg in args_list]

    t, traj, x, dxdt = duffing_osc_i(args)
    
    return t, traj, x, dxdt


alpha = 1.0
beta = 5.0
gamma = 8.0
delta = 0.02
omega = 0.5

args_list = [alpha,beta,gamma,delta,omega]

tlist, traj, x, dxdt = duffing_osc(args_list)

w_demod = omega

demod1 = (traj*np.exp(1j*w_demod*tlist))

limits = np.max(np.abs(demod1))*2

fig1, ax1 = plt.subplots(figsize=(8,5))
fig2, ax2 = plt.subplots(figsize=(8,5))
fig3, ax3 = plt.subplots(figsize=(8,5))

ax1.plot(np.real(demod1), np.imag(demod1))
ax1.scatter(np.real(demod1[-1]), np.imag(demod1[-1]), label="1")
ax1.legend()
ax1.set_title('Vacuum Rabi oscillations')
ax1.set_xlim([-limits,limits])
ax1.set_ylim([-limits,limits])

ax2.plot(tlist, x)
ax2.plot(tlist, dxdt)
ax2.set_xlabel('Time')
ax2.set_xlabel('Time')

fft_out = np.fft.fft(x)

freq = np.linspace(0,len(tlist)/tlist[-1],len(tlist))

ax3.plot(freq, np.log(np.abs(fft_out)))