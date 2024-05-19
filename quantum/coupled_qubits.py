import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

N = 5

q1 = qt.destroy(5)

q1*q1.dag()

ω1s = np.linspace(5e9*2*np.pi,6e9*2*np.pi,100)

N = 5

eigenenergies = np.zeros([N, len(ω1s)])

for i in range(0,len(ω1s)):

    ω1 = 5e9*2*np.pi

    α1 = -1e9*2*np.pi

    q1 = qt.destroy(N)

    H1 = ω1s[i]*q1*q1.dag() + α1/2*(q1*q1.dag())**2
    
    eigenenergies[:,i] = H1.eigenenergies()
    
plt.plot(ω1s,eigenenergies.transpose(),'k')

#%%

N = 5

ω1 = 5e9*2*np.pi
ω2 = 6e9*2*np.pi

α1 = -100e6*2*np.pi
α2 = -100e6*2*np.pi

g = 100e6*2*np.pi

N = 3

q1 = qt.tensor(qt.destroy(N),qt.qeye(N))
q2 = qt.tensor(qt.qeye(N),qt.destroy(N))

eigenenergies = np.zeros([N**2, len(ω1s)])

ω1s = np.linspace(5e9*2*np.pi,6e9*2*np.pi,100)

for i in range(0,len(ω1s)):

    ω1 = 5e9*2*np.pi

    α1 = -1e9*2*np.pi

    H1 = ω1s[i]*q1*q1.dag() + α1/2*(q1*q1.dag())**2
    H2 = ω2*q2*q2.dag() + α2/2*(q2*q2.dag())**2
    Hcouple = g*(q1 + q1.dag())*(q2 + q2.dag())
    
    H = H1 + H2 + Hcouple
    
    eigenenergies[:,i] = H.eigenenergies()
    
plt.plot(ω1s,eigenenergies.transpose(),'k')