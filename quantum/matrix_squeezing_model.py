import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter


'''
This is a description of one-mode squeezed states of light in a 2-dimensional space
'''

def apply_transformation(M, PDF, i1, q1):
    
    I1, Q1 = np.meshgrid(i1, q1, indexing='ij')
    
    Minv = np.linalg.inv(M)
    
    I1T = Minv[0,0]*I1 + Minv[0,1]*Q1
    Q1T = Minv[1,0]*I1 + Minv[1,1]*Q1
    
    
    points = (i1, q1)
    values = PDF.copy()

    xi = np.array([I1T.flatten(),Q1T.flatten()]).transpose()
    
    
    results = interpn(points, values, xi, bounds_error=False, fill_value=0)
    
    results = np.reshape(results, np.shape(I1))
    
    return results 


r = 1
phi = np.pi*0
oneMS = np.array([[np.cosh(r)+np.cos(phi)*np.sinh(r), np.sin(phi)*np.sinh(r)],
                  [np.sin(phi)*np.sinh(r), np.cosh(r)-np.sinh(r)*np.cos(phi)]])

theta = np.pi*0
rotation = np.array([[np.cos(theta), np.sin(theta)],
                     [-np.sin(theta), np.cos(theta)]])

G = 0.5
attenuate = np.array([[G,0],[0,G]])

# histogram

i1 = np.linspace(-2,1,501)
q1 = np.linspace(-1,1,501)

I1, Q1 = np.meshgrid(i1, q1, indexing='ij')

i1_0 = 0
q1_0 = 0.1

sigma = 0.05

PDF = np.exp( -((I1-i1_0)**2 + (Q1-q1_0)**2) /sigma**2)

plt.figure(1)
plt.clf()
plt.pcolor(i1, q1, PDF.transpose())

PDF = apply_transformation(oneMS, PDF, i1, q1)
PDF = apply_transformation(attenuate, PDF, i1, q1)

plt.figure(2)
plt.clf()
plt.pcolor(i1, q1, PDF.transpose())

PDF = gaussian_filter(PDF, 10.1, mode='nearest')

plt.figure(3)
plt.clf()
plt.pcolor(i1, q1, PDF.transpose())


#%%

'''
This is a description of two-mode squeezed states of light in a 4-dimensional space, corresponding 
to the two components of signal and idler modes
'''


def apply_transformation(M, PDF, i1, q1, i2, q2):
    
    I1, Q1, I2, Q2 = np.meshgrid(i1, q1, i2, q2, indexing='ij')
    
    Minv = np.linalg.inv(M)
    
    I1T = Minv[0,0]*I1 + Minv[0,1]*Q1 + Minv[0,2]*I2 + Minv[0,3]*Q2
    Q1T = Minv[1,0]*I1 + Minv[1,1]*Q1 + Minv[1,2]*I2 + Minv[1,3]*Q2
    I2T = Minv[2,0]*I1 + Minv[2,1]*Q1 + Minv[2,2]*I2 + Minv[2,3]*Q2
    Q2T = Minv[3,0]*I1 + Minv[3,1]*Q1 + Minv[3,2]*I2 + Minv[3,3]*Q2
    
    points = (i1, q1, i2, q2)
    values = PDF.copy()

    xi = np.array([I1T.flatten(),Q1T.flatten(),I2T.flatten(),Q2T.flatten()]).transpose()
    
    results = interpn(points, values, xi, bounds_error=False, fill_value=0)
    
    results = np.reshape(results, np.shape(I1))
    
    return results 


theta = np.pi*0.25
signal_rotation = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                            [np.sin(theta), np.cos(theta), 0, 0],
                            [0,0,1,0],
                            [0,0,0,1]])

r = 1
phi = np.pi*0

theta = 0
twoMS = np.array([
            [np.cosh(r), 0, np.cos(phi) * np.sinh(r), -np.sin(phi) * np.sinh(r)],
            [0, np.cosh(r), np.sin(phi) * np.sinh(r), -np.cos(phi) * np.sinh(r)],
            [np.cos(phi) * np.sinh(r), -np.sin(phi) * np.sinh(r), np.cosh(r), 0],
            [np.sin(phi) * np.sinh(r), -np.cos(phi) * np.sinh(r), 0, np.cosh(r)]])


i1 = np.linspace(-1,1,41)
q1 = np.linspace(-1,1,41)
i2 = np.linspace(-1,1,41)
q2 = np.linspace(-1,1,41)

I1, Q1, I2, Q2 = np.meshgrid(i1, q1, i2, q2, indexing='ij')

i1_0 = 0.5
q1_0 = 0
i2_0 = 0
q2_0 = 0


sigma = 0.1

PDF = np.exp( -((I1-i1_0)**2 + (Q1-q1_0)**2 + (I2-i2_0)**2 + (Q2-q2_0)**2) /sigma**2)

plt.figure(1)
plt.clf()
plt.pcolor(i1, q1, np.mean(PDF,axis=(2,3)).transpose())


G = 0.2
attenuate = np.eye(4)*G

PDF = apply_transformation(twoMS, PDF, i1, q1, i2, q2)
# PDF = apply_transformation(attenuate, PDF, i1, q1)

#%%

plt.figure()
plt.clf()
plt.pcolor(i1, i2, np.mean(PDF,axis=(1,3)).transpose())

#%%

PDF = gaussian_filter(PDF, 1, mode='nearest')

plt.figure(3)
plt.clf()
plt.pcolor(i1, q1, np.mean(PDF,axis=(2,3)).transpose())

#%%

theta = np.pi*0
signal_rotation = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                            [np.sin(theta), np.cos(theta), 0, 0],
                            [0,0,1,0],
                            [0,0,0,1]])

r = 1
phi = np.pi*0.25


twoMS = np.array([
            [np.cosh(r), 0, np.cos(phi) * np.sinh(r), -np.sin(phi) * np.sinh(r)],
            [0, np.cosh(r), np.sin(phi) * np.sinh(r), -np.cos(phi) * np.sinh(r)],
            [np.cos(phi) * np.sinh(r), -np.sin(phi) * np.sinh(r), np.cosh(r), 0],
            [np.sin(phi) * np.sinh(r), -np.cos(phi) * np.sinh(r), 0, np.cosh(r)]])


i1 = np.linspace(-1,1,41)
q1 = np.linspace(-1,1,41)
i2 = np.linspace(-1,1,41)
q2 = np.linspace(-1,1,41)

I1, Q1, I2, Q2 = np.meshgrid(i1, q1, i2, q2, indexing='ij')

i1_0 = 0.4
q1_0 = 0
i2_0 = 0
q2_0 = 0


sigma = 0.2

PDF1 = np.exp( -((I1-i1_0)**2 + (Q1-q1_0)**2 + (I2-i2_0)**2 + (Q2-q2_0)**2) /sigma**2)
PDF1 = PDF1/np.sum(PDF1)
PDF2 = apply_transformation(signal_rotation, PDF1, i1, q1, i2, q2)

PDF1_s = np.sum(PDF1,axis=(2,3))
PDF2_s = np.sum(PDF2,axis=(2,3))

PDF = PDF1-PDF2

PDF_s = np.mean(PDF,axis=(2,3))

plt.figure()
plt.clf()
plt.pcolor(i1, q1, np.minimum(PDF1_s,PDF2_s).transpose())

print('Fidelity: ' + str( 1- np.sum(np.minimum(PDF1_s,PDF2_s))/2 ) )
      

#%%
plt.figure()
plt.clf()

PDF1_I = np.sum(PDF1_s, axis=1)
PDF2_I = np.sum(PDF2_s, axis=1)

plt.plot(i1, PDF1_I)

plt.plot(i1, PDF2_I)

plt.plot(i1, np.min([PDF1_I,PDF2_I],axis=0))

#%% for real now


def make_signalrotation_op(theta):
    signal_rotation = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                                [np.sin(theta), np.cos(theta), 0, 0],
                                [0,0,1,0],
                                [0,0,0,1]])
    
    return signal_rotation


def make_TMS_op(r, phi):
    
    r = 1
    phi = np.pi*0.25
    twoMS = np.array([
                [np.cosh(r), 0, np.cos(phi) * np.sinh(r), -np.sin(phi) * np.sinh(r)],
                [0, np.cosh(r), np.sin(phi) * np.sinh(r), -np.cos(phi) * np.sinh(r)],
                [np.cos(phi) * np.sinh(r), -np.sin(phi) * np.sinh(r), np.cosh(r), 0],
                [np.sin(phi) * np.sinh(r), -np.cos(phi) * np.sinh(r), 0, np.cosh(r)]])
    
    return twoMS

r = 1
phi = np.pi*0.25
twoMS2 = np.array([
            [np.cosh(r), 0, np.cos(phi) * np.sinh(r), -np.sin(phi) * np.sinh(r)],
            [0, np.cosh(r), np.sin(phi) * np.sinh(r), -np.cos(phi) * np.sinh(r)],
            [np.cos(phi) * np.sinh(r), -np.sin(phi) * np.sinh(r), np.cosh(r), 0],
            [np.sin(phi) * np.sinh(r), -np.cos(phi) * np.sinh(r), 0, np.cosh(r)]])

