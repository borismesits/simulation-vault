import numpy as np
import matplotlib.pyplot as plt




def R(sample, theta):
    
    return sample*np.exp(-1j*theta)

def D(sample, disp):
    
    return sample + disp

def S(sample, r, phi):
    
    return np.cosh(r)*sample + np.exp(-1j*phi)*np.sinh(r)*np.conjugate(sample)
    

t = np.linspace(0,1e-7,10001)
dt = t[1]-t[0]

for i in range(0,1000):

    x = np.zeros(len(t), dtype=np.complex128) # for two modes
    
    x[0] = np.random.normal(0,1)+1j*np.random.normal(0,1)
    
    chi = 1e5
    
    for i in range(0,len(t)-1):
        
        b = x[i]
        
        b1 = R(b, 2*chi*np.abs(b)**2*dt)
        b2 = D(b1, chi*np.abs(b)**2*np.conjugate(b)*dt)
        b3 = S(b2, 0.5*chi*b**2*dt,0)
    
        x[i+1] = b3
    
        # x[i+1] = S(b, 0.5*chi*b**2*dt,0)
        # x[i+1] = D(b, chi*np.abs(b)**2*np.conjugate(b)*dt)
        # x[i+1] = R(b, 2*chi*np.abs(b)**2*dt)
        
    plt.plot(np.real(x), np.imag(x))