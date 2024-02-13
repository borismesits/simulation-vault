import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,5,10000)

omega_0 = 10*2*np.pi

C = 1

C_mod = 0.01

phi_mod = 0

phi = np.pi*0.5

omega_mod = 0.2*np.pi*2

noise_amp  = 5e-1

A0 = C*np.sin(omega_0*t) + np.random.normal(0, noise_amp, size=len(t))

A1 = C*np.sin(omega_0*t + phi) * (1 + C_mod*np.sin(omega_mod*t + phi_mod)) + np.random.normal(0, noise_amp, size=len(t))

A2 = C*np.sin(omega_0*t + phi + (C_mod*np.sin(omega_mod*t + phi_mod))*omega_0 ) + np.random.normal(0, noise_amp, size=len(t))

D = A2 + A0 # for detected, or digitized

fig,axs = plt.subplots(3,1)

def decimate(signal, omega, t, factor):
    
    signal_rs = np.reshape(signal*np.sin(omega*t), (len(signal)//factor, factor))
    
    signal_dc = np.mean(signal_rs, axis=1)
    
    t_rs = np.reshape(t, (len(t)//factor, factor))
    
    t_dc = np.mean(t_rs, axis=1)
    
    return t_dc, signal_dc

factor = 100

t_dc, D_dc = decimate(D, omega_0, t, factor)

axs[0].plot(t, D)
axs[1].plot(t_dc, D_dc)
axs[2].plot(t, D-A0)
axs[2].plot(t, A0)

