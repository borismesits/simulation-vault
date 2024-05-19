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
one mode, ain. 

ain is described by Gaussian noise of variance sigma and average (Iavg, Qavg)

Squeezing axis is theta, gain is G

Note the effect on the noise causes "squished" noise, i.e., correlated
'''

sigma = 0.3

Iavg = 0
Qavg = 2

ain_R = np.random.normal(Iavg,sigma,size=1000)
ain_I = np.random.normal(Qavg,sigma,size=1000)

ain = ain_R + 1j*ain_I

ain = apply_rotation(np.pi/4, ain)

theta = np.pi

G = 5

r = np.arccosh(np.sqrt(G))

bout = np.cosh(r)*ain + np.exp(1j*theta)*np.sinh(r)*np.conj(ain)

plt.figure()
plt.scatter(np.real(ain),np.imag(ain))
plt.scatter(np.real(bout),np.imag(bout))
plt.xlim([-10,10])
plt.ylim([-10,10])

#%%

'''
Nonegenerate parametric amplifier, from Barzanjeh eq. 8. Signal and idler are separate, so input ain and outbout bout 
are vectors of size 2.

If you set theta=0, the noise correlation is really easy to see. However, we don't see the "squished" shape.
'''

#theta = np.pi/2
theta = 0

G = 100

r = np.arccosh(np.sqrt(G))

S = np.array([[np.cosh(r), np.exp(1j*theta)*np.sinh(r)],[np.exp(-1j*theta)*np.sinh(r), np.cosh(r)]])

ain1_R = np.random.normal(0,0.3,size=1000)
ain1_I = np.random.normal(0,0.3,size=1000)

ain1 = ain1_R + 1j*ain1_I

ain2_R = np.random.normal(0,0.3,size=1000)
ain2_I = np.random.normal(0,0.3,size=1000)

ain2 = ain2_R + 1j*ain2_I

ain = np.array([ain1, ain2])

bout = np.dot(S, ain)



plt.scatter(np.real(ain[0]),np.imag(ain[0])) # plot points in IQ plane, for both components of ain and bout
plt.scatter(np.real(ain[1]),np.imag(ain[1]))

plt.scatter(np.real(bout[0]),np.imag(bout[0]))
plt.scatter(np.real(bout[1]),np.imag(bout[1]))

plt.xlim([-10,10])
plt.ylim([-10,10])

#%%

'''
Even with two-mode squeezing (nondegenerate gain), it's possible to see the squishing effect.

All you have to do is plot the I component - i.e., np.real() - of the signal and the I component of the idler.

Before we were ploting the real and imaginary of signal OR idler - NOT mixing the two
'''

plt.scatter(np.real(ain[0]),np.real(ain[1]))
plt.scatter(np.real(bout[0]),np.real(bout[1]))



#%%


'''
Nonegenerate parametric amplifier, from Barzanjeh eq. 8. Signal and idler are separate, so input ain and outbout bout 
are vectors of size 2.

If you set theta=0, the noise correlation is really easy to see. However, we don't see the "squished" shape.
'''


def apply_PP_squeezing(si, G, theta):
    
    r = np.arccosh(np.sqrt(G))
    
    S = np.array([[np.cosh(r), np.exp(1j*theta)*np.sinh(r)],[np.exp(-1j*theta)*np.sinh(r), np.cosh(r)]])

    return np.dot(S, si)

sigma = 0.1

a1_R = np.random.normal(0,sigma,size=1000)
a1_I = np.random.normal(1,sigma,size=1000)

a1 = a1_R + 1j*a1_I

a2_R = np.random.normal(0,sigma,size=1000)
a2_I = np.random.normal(0,sigma,size=1000)

a2 = a2_R + 1j*a2_I

a = np.array([a1, a2])

phi = np.pi*0.05
# a = np.array([apply_rotation(phi, a[0]),a[1]])

G = 5
theta = np.pi*0.5
b = apply_PP_squeezing(a, G, theta)

phi = np.pi*0.1

c = np.array([apply_rotation(phi, b[0]),b[1]])

G = 5
theta = -np.pi*0.5
d = apply_PP_squeezing(c, G, theta) 

fig, axs = plt.subplots(nrows=1,ncols=2)

axs[0].scatter(np.real(a[0]),np.imag(a[0]),color='r',marker='_') # plot points in IQ plane, for both components of ain and bout
axs[1].scatter(np.real(a[1]),np.imag(a[1]),color='r',marker='|')

axs[0].scatter(np.real(b[0]),np.imag(b[0]),color='g',marker='_')
axs[1].scatter(np.real(b[1]),np.imag(b[1]),color='g',marker='|')

axs[0].scatter(np.real(c[0]),np.imag(c[0]),color='b',marker='_')
axs[1].scatter(np.real(c[1]),np.imag(c[1]),color='b',marker='|')

axs[0].scatter(np.real(d[0]),np.imag(d[0]),color='k',marker='_')
axs[1].scatter(np.real(d[1]),np.imag(d[1]),color='k',marker='|')

axs[0].set_xlim([-10,10])
axs[0].set_ylim([-10,10])

axs[1].set_xlim([-10,10])
axs[1].set_ylim([-10,10])

axs[0].grid()
axs[1].grid()