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


plt.figure()
for i in range(0,300):

    x = np.zeros(len(t), dtype=np.complex128) # for two modes
    
    noise = np.random.rand() + 1j*np.random.rand()
    
    x[0] = 10 + noise*0.5

    chi = 10e4
    
    for i in range(0,len(t)-1):
        
        b = x[i]
        
        b1 = R(b, 2*chi*np.abs(b)**2*dt)
        b2 = D(b1, chi*np.abs(b)**2*b*dt)
        b3 = S(b2, 0.5*chi*np.abs(b)**2*dt,0)
        
        # b1 = S(b, 0.5*chi*np.abs(b)**2*dt,0)
        # b2 = D(b1, chi*np.abs(b)**2*b*dt)
        # b3 = R(b2, 2*chi*np.abs(b)**2*dt)
    
        x[i+1] = b3
    
        # x[i+1] = S(b, 0.5*chi*np.abs(b)**2*dt,0)
        # x[i+1] = D(b, chi*np.abs(b)**2*b*dt)
        # x[i+1] = R(b, 2*chi*np.abs(b)**2*dt)
        
    plt.scatter(np.real(x[0]), np.imag(x[0]),color='r')
    plt.scatter(np.real(x[-1]), np.imag(x[-1]),color='b')
    