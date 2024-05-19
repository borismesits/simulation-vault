import numpy as np
import matplotlib.pyplot as plt


def apply_rotation(angle, vector):
    
    N = len(vector)
    
    phi = np.ones(N)*angle
    phi[0:N//2] = -angle
    
    vector_new = vector*np.exp(phi*1j)
    
    return vector_new
        
#%%

'''
Degenerate parametric amplifier, from Barzanjeh eq. 7. Signal and idler overlap, so we have only 
one mode, a. 

a is described by Gaussian noise of variance sigma and average (Iavg, Qavg)

Squeezing axis is theta, gain is G

Note the effect on the noise causes "squished" noise, i.e., correlated
'''

sigma = 0.3
Iavg = 0
Qavg = 0

a_R = np.random.normal(Iavg,sigma,size=1000)
a_I = np.random.normal(Qavg,sigma,size=1000)

a = a_R + 1j*a_I


theta = 0

G = 2
r = np.arccosh(np.sqrt(G))
print(r)
b = np.cosh(r)*a + np.exp(1j*theta)*np.sinh(r)*np.conj(a)

G = 2
r = np.arccosh(np.sqrt(G))
print(r)
c = np.cosh(r)*b + np.exp(1j*theta)*np.sinh(r)*np.conj(b)


plt.figure()
plt.scatter(np.real(a),np.imag(a))
plt.scatter(np.real(b),np.imag(b))
plt.scatter(np.real(c),np.imag(c))
plt.xlim([-10,10])
plt.ylim([-10,10])

cov = np.sqrt(np.cov(np.real(c), np.imag(c)))
SF = cov[0,0]/cov[1,1]



print(SF)


r_expt = np.log(SF)/2

print(r_expt)




#%%


r = np.linspace(0,5)
SF = (np.cosh(r)+np.sinh(r))/(np.cosh(r)-np.sinh(r))
plt.plot(r, np.log10(np.exp(r*2)))
plt.plot(r,np.log10(SF))