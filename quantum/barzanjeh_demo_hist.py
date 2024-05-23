import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

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

Iavg = 1
Qavg = 0

N = 1000000

ain_R = np.random.normal(Iavg,sigma,size=N)
ain_I = np.random.normal(Qavg,sigma,size=N)

ain = ain_R + 1j*ain_I

ain = apply_rotation(np.pi/4, ain)

theta = np.pi

G = 2

r = np.arccosh(np.sqrt(G))

bout = np.cosh(r)*ain + np.exp(1j*theta)*np.sinh(r)*np.conj(ain)

boutg = bout[0:N//2]
boute = bout[N//2:]

hist_range = 5

bhistg, x, y = np.histogram2d(np.real(boutg), np.imag(boutg), bins=201, range=[[-hist_range, hist_range],[-hist_range, hist_range]])
bhiste, x, y = np.histogram2d(np.real(boute), np.imag(boute), bins=201, range=[[-hist_range, hist_range],[-hist_range, hist_range]])

img = bhistg/np.max(bhistg) - bhiste/np.max(bhiste)
total = bhistg/np.max(bhistg) + bhiste/np.max(bhiste)


r = img*0
g = 0.5-img/4
b = 0.5+img/2
a = total

rgba = np.stack([r,g,b,a],axis=-1)
rgba = np.flip(rgba,axis=0)


plt.imshow(rgba,extent=[-hist_range, hist_range,-hist_range, hist_range])
plt.plot([0,np.mean(np.imag(boutg))],[0,np.mean(np.real(boutg))],'k--')
plt.plot([0,np.mean(np.imag(boute))],[0,np.mean(np.real(boute))],'k--')
plt.clim([-1.0,1.0])
plt.axis('off')
ax = plt.gca()
ax.set_aspect('equal')
plt.savefig('demo.png', transparent=True)
#%%

'''
Nonegenerate parametric amplifier, from Barzanjeh eq. 8. Signal and idler are separate, so input ain and outbout bout 
are vectors of size 2.

If you set theta=0, the noise correlation is really easy to see. However, we don't see the "squished" shape.
'''

#theta = np.pi/2
theta = 0
 
G = 4

r = np.arccosh(np.sqrt(G))

S = np.array([[np.cosh(r), np.exp(1j*theta)*np.sinh(r)],[np.exp(-1j*theta)*np.sinh(r), np.cosh(r)]])

ain1_R = np.random.normal(Iavg,sigma,size=N)
ain1_I = np.random.normal(Qavg,sigma,size=N)

ain1 = ain1_R + 1j*ain1_I

ain2_R = np.random.normal(0,sigma,size=N)
ain2_I = np.random.normal(0,sigma,size=N)

ain2 = ain2_R + 1j*ain2_I

ain = np.array([apply_rotation(np.pi/4, ain1), ain2])


bout = np.dot(S, ain)



boutg = bout[0,0:N//2]
boute = bout[0,N//2:]

hist_range = 5

bhistg, x, y = np.histogram2d(np.real(boutg), np.imag(boutg), bins=201, range=[[-hist_range, hist_range],[-hist_range, hist_range]])
bhiste, x, y = np.histogram2d(np.real(boute), np.imag(boute), bins=201, range=[[-hist_range, hist_range],[-hist_range, hist_range]])

img = bhistg/np.max(bhistg) - bhiste/np.max(bhiste)
total = bhistg/np.max(bhistg) + bhiste/np.max(bhiste)


r = img*0
g = 0.5-img/4
b = 0.5+img/2
a = total

rgba = np.stack([r,g,b,a],axis=-1)
rgba = np.flip(rgba,axis=0)


plt.imshow(rgba,extent=[-hist_range, hist_range,-hist_range, hist_range])
plt.plot([0,np.mean(np.imag(boutg))],[0,np.mean(np.real(boutg))],'k--')
plt.plot([0,np.mean(np.imag(boute))],[0,np.mean(np.real(boute))],'k--')
plt.clim([-1.0,1.0])
plt.axis('off')
ax = plt.gca()
ax.set_aspect('equal')
plt.savefig('demo.png', transparent=True)
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

n_in = 10

sigma = np.sqrt(n_in)

a1_R = np.random.normal(n_in,sigma/np.sqrt(2),size=N)
a1_I = np.random.normal(0,sigma/np.sqrt(2),size=N)

a1 = a1_R + 1j*a1_I

a2_R = np.random.normal(0,sigma/np.sqrt(2),size=N)
a2_I = np.random.normal(0,sigma/np.sqrt(2),size=N)

a2 = a2_R + 1j*a2_I

a = np.array([a1, a2])

# phi = np.pi*0.4
# a = np.array([apply_rotation(phi, a[0]),a[1]])

G_1 = 10
G_2 = 10

offset_theta = np.pi*0.0

theta = offset_theta
b = apply_PP_squeezing(a, G_1, theta)

phi = np.pi*0.1
c = np.array([apply_rotation(phi, b[0]),b[1]])

theta = np.pi + offset_theta
d = apply_PP_squeezing(c, G_2, theta)

# plt.scatter(np.real(a[0]),np.imag(a[0]),color='r',marker='_') # plot points in IQ plane, for both components of ain and bout
# plt.scatter(np.real(a[1]),np.imag(a[1]),color='r',marker='|')

# plt.scatter(np.real(b[0]),np.imag(b[0]),color='g',marker='_')
# plt.scatter(np.real(b[1]),np.imag(b[1]),color='g',marker='|')

# plt.scatter(np.real(c[0]),np.imag(c[0]),color='b',marker='_')
# plt.scatter(np.real(c[1]),np.imag(c[1]),color='b',marker='|')

# plt.scatter(np.real(d[0]),np.imag(d[0]),color='k',marker='_')
# plt.scatter(np.real(d[1]),np.imag(d[1]),color='k',marker='|')

# plt.xlim([-100,100])
# plt.ylim([-100,100])

a_s_g_mean = np.mean(a[0,0:N//2])
a_s_g_std = np.std(a[0,0:N//2])
a_s_e_mean = np.mean(a[0,N//2:])
a_s_e_std = np.std(a[0,N//2:])
a_i_g_mean = np.mean(a[1,0:N//2])
a_i_g_std = np.std(a[1,0:N//2])
a_i_e_mean = np.mean(a[1,N//2:])
a_i_e_std = np.std(a[1,N//2:])

print('A signal g: ' + str(np.round(a_s_g_mean,4)) + ' , std: ' + str(np.round(a_s_g_std,4)))
print('A signal e: ' + str(np.round(a_s_e_mean,4)) + ' , std: ' + str(np.round(a_s_e_std,4)))
print('A idler g: ' + str(np.round(a_i_g_mean,4)) + ' , std: ' + str(np.round(a_i_g_std,4)))
print('A idler e: ' + str(np.round(a_i_e_mean,4)) + ' , std: ' + str(np.round(a_i_e_std,4)))


b_s_g_mean = np.mean(b[0,0:N//2])
b_s_g_std = np.std(b[0,0:N//2])
b_s_e_mean = np.mean(b[0,N//2:])
b_s_e_std = np.std(b[0,N//2:])
b_i_g_mean = np.mean(b[1,0:N//2])
b_i_g_std = np.std(b[1,0:N//2])
b_i_e_mean = np.mean(b[1,N//2:])
b_i_e_std = np.std(b[1,N//2:])

print('B signal g: ' + str(np.round(b_s_g_mean,4)) + ' , std: ' + str(np.round(b_s_g_std,4)))
print('B signal e: ' + str(np.round(b_s_e_mean,4)) + ' , std: ' + str(np.round(b_s_e_std,4)))
print('B idler g: ' + str(np.round(b_i_g_mean,4)) + ' , std: ' + str(np.round(b_i_g_std,4)))
print('B idler e: ' + str(np.round(b_i_e_mean,4)) + ' , std: ' + str(np.round(b_i_e_std,4)))


c_s_g_mean = np.mean(c[0,0:N//2])
c_s_g_std = np.std(c[0,0:N//2])
c_s_e_mean = np.mean(c[0,N//2:])
c_s_e_std = np.std(c[0,N//2:])
c_i_g_mean = np.mean(c[1,0:N//2])
c_i_g_std = np.std(c[1,0:N//2])
c_i_e_mean = np.mean(c[1,N//2:])
c_i_e_std = np.std(c[1,N//2:])

print('C signal g: ' + str(np.round(c_s_g_mean,4)) + ' , std: ' + str(np.round(c_s_g_std,4)))
print('C signal e: ' + str(np.round(c_s_e_mean,4)) + ' , std: ' + str(np.round(c_s_e_std,4)))
print('C idler g: ' + str(np.round(c_i_g_mean,4)) + ' , std: ' + str(np.round(c_i_g_std,4)))
print('C idler e: ' + str(np.round(c_i_e_mean,4)) + ' , std: ' + str(np.round(c_i_e_std,4)))


d_s_g_mean = np.mean(d[0,0:N//2])
d_s_g_std = np.std(d[0,0:N//2])
d_s_e_mean = np.mean(d[0,N//2:])
d_s_e_std = np.std(d[0,N//2:])
d_i_g_mean = np.mean(d[1,0:N//2])
d_i_g_std = np.std(d[1,0:N//2])
d_i_e_mean = np.mean(d[1,N//2:])
d_i_e_std = np.std(d[1,N//2:])

print('D signal g: ' + str(np.round(d_s_g_mean,4)) + ' , std: ' + str(np.round(d_s_g_std,4)))
print('D signal e: ' + str(np.round(d_s_e_mean,4)) + ' , std: ' + str(np.round(d_s_e_std,4)))
print('D idler g: ' + str(np.round(d_i_g_mean,4)) + ' , std: ' + str(np.round(d_i_g_std,4)))
print('D idler e: ' + str(np.round(d_i_e_mean,4)) + ' , std: ' + str(np.round(d_i_e_std,4)))


a_s_snr = np.abs(a_s_g_mean-a_s_e_mean)/(np.mean([a_s_g_std,a_s_g_std]))
a_i_snr = np.abs(a_i_g_mean-a_i_e_mean)/(np.mean([a_i_g_std,a_i_g_std]))

b_s_snr = np.abs(b_s_g_mean-b_s_e_mean)/(np.mean([b_s_g_std,b_s_g_std]))
b_i_snr = np.abs(b_i_g_mean-b_i_e_mean)/(np.mean([b_i_g_std,b_i_g_std]))

c_s_snr = np.abs(c_s_g_mean-c_s_e_mean)/(np.mean([c_s_g_std,c_s_g_std]))
c_i_snr = np.abs(c_i_g_mean-c_i_e_mean)/(np.mean([c_i_g_std,c_i_g_std]))

d_s_snr = np.abs(d_s_g_mean-d_s_e_mean)/(np.mean([d_s_g_std,d_s_g_std]))
d_i_snr = np.abs(d_i_g_mean-d_i_e_mean)/(np.mean([d_i_g_std,d_i_g_std]))


print('A SNR signal: ' + str(np.round(a_s_snr,4)))
print('A SNR idler: ' + str(np.round(a_i_snr,4)))

print('B SNR signal: ' + str(np.round(b_s_snr,4)))
print('B SNR idler: ' + str(np.round(b_i_snr,4)))

print('C SNR signal: ' + str(np.round(c_s_snr,4)))
print('C SNR idler: ' + str(np.round(c_i_snr,4)))

print('D SNR signal: ' + str(np.round(d_s_snr,4)))
print('D SNR idler: ' + str(np.round(d_i_snr,4)))

A_N_1 = 0.5*(1 - 1/G_1)
A_N_2 = 0.5*(1 - 1/G_2)
coherentPA = 2*np.sqrt(n_in)*np.abs(np.sin(phi))/np.sqrt(2*A_N_2+1)

SU11PA = 2*np.sqrt(n_in)*np.abs(np.sin(phi))/np.sqrt( (2*A_N_1+1)*(2*A_N_2+1) - 8*np.cos(phi)*np.sqrt(A_N_1*A_N_2))

total_gain = np.abs(d_s_g_mean)**2/np.abs(a_s_g_mean)**2

print('SNR coherent+PA: ' + str(np.round(coherentPA,4)))
print('SNR SU11+PA: ' + str(np.round(SU11PA,4)))
print('Total gain: ' + str(np.round(total_gain,3)))


#%%



def make_tms_hists(mode, mode_name, hist_range, axs, i):
    
    signal_g = mode[0,0:N//2]
    signal_e = mode[0,N//2:]
    
    idler_g = mode[1,0:N//2]
    idler_e = mode[1,N//2:]

    def make_hists(Ig, Ie, Qg, Qe, name, axs, j):
    
        hist_g, x, y = np.histogram2d(Ig, Qg, bins=201, range=[[-hist_range, hist_range],[-hist_range, hist_range]])
        hist_e, x, y = np.histogram2d(Ie, Qe, bins=201, range=[[-hist_range, hist_range],[-hist_range, hist_range]])
    
    
        img = hist_g/np.max(hist_g) - hist_e/np.max(hist_e)
        total = hist_g/np.max(hist_e) + hist_e/np.max(hist_e)
    
        G = img*0
        R = 0.5-img/4
        B = 0.5+img/2
        A = total
        
        rgba = np.stack([R,G,B,A],axis=-1)
        rgba = np.flip(rgba,axis=0)
    
        axs[j,i].imshow(rgba,extent=[-hist_range, hist_range,-hist_range, hist_range],interpolation='bicubic')
        axs[j,i].plot([-hist_range, hist_range],[0,0],'k',linewidth=0.5)
        axs[j,i].plot([0,0],[-hist_range, hist_range],'k',linewidth=0.5)
        # axs[i,j].set_clim([-1.0,1.0])
        # axs[j,i].set_title('asdfd')
        axs[j,i].axis('off')
 
        axs[j,i].set_aspect('equal')
        
        
    make_hists(np.real(signal_g), np.real(signal_e), np.imag(signal_g), np.imag(signal_e), 'signal', axs, 0)
    make_hists(np.real(idler_g), np.real(idler_e), np.imag(idler_g), np.imag(idler_e), 'idler', axs, 1)
    make_hists(np.real(signal_g), np.real(signal_e), np.real(idler_g), np.real(idler_e), 'cross', axs, 2)

fig, axs = plt.subplots(nrows=3,ncols=4)

make_tms_hists(a,'a', 70, axs, 0)
make_tms_hists(b,'b', 70, axs, 1)    
make_tms_hists(c,'c', 70, axs, 2)
make_tms_hists(d,'d', 70, axs, 3)
plt.tight_layout()

fig.savefig('full_demo.png', transparent=True, figsize=(7,4))