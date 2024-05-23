import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter
import cupy as cp


def make_signalrotation_op(theta):
    signal_rotation = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                                [np.sin(theta), np.cos(theta), 0, 0],
                                [0,0,1,0],
                                [0,0,0,1]])
    
    return signal_rotation


def make_TMS_op(r, phi):

    twoMS = np.array([
                [np.cosh(r), 0, np.cos(phi) * np.sinh(r), -np.sin(phi) * np.sinh(r)],
                [0, np.cosh(r), np.sin(phi) * np.sinh(r), np.cos(phi) * np.sinh(r)],
                [np.cos(phi) * np.sinh(r), np.sin(phi) * np.sinh(r), np.cosh(r), 0],
                [-np.sin(phi) * np.sinh(r), np.cos(phi) * np.sinh(r), 0, np.cosh(r)]])
    
    return twoMS

def make_OMS_op(r, phi):

    oneMS = np.array([[np.cosh(r)+np.cos(phi)*np.sinh(r), np.sin(phi)*np.sinh(r),0,0],
                      [np.sin(phi)*np.sinh(r), np.cosh(r)-np.sinh(r)*np.cos(phi),0,0],
                      [0,0,np.cosh(r)+np.cos(phi)*np.sinh(r), np.sin(phi)*np.sinh(r)],
                      [0,0,np.cosh(r)+np.cos(phi)*np.sinh(r), np.sin(phi)*np.sinh(r)]])
    
    return oneMS

def OMS_exp(N, nbar=1, phi_r=np.pi/2, theta=np.pi, noise=0, r1=0, r2=0, phi1=0, phi2=0, plot=True):
    
    sigma = 0.4
    
    # define ops
    signal_rotation_g = cp.array(make_signalrotation_op(theta/2))
    signal_rotation_e = cp.array(make_signalrotation_op(-theta/2))
    OMS1 = cp.array(make_OMS_op(r1, phi1))
    OMS2 = cp.array(make_OMS_op(r2, phi2))
    
    sampg = cp.random.normal(0, sigma, size=(4,N)) # create random samples
    sampe = cp.random.normal(0, sigma, size=(4,N))
    
    sampg += cp.expand_dims(cp.array([nbar,0,0,0]),1) # displacement
    sampe += cp.expand_dims(cp.array([nbar,0,0,0]),1)
    
    sampg = cp.dot(OMS1,sampg) # TMS 1
    sampe = cp.dot(OMS1,sampe)
    
    sampg = cp.dot(signal_rotation_g,sampg) # dispersive shift
    sampe = cp.dot(signal_rotation_e,sampe)
    
    sampg = cp.dot(OMS2,sampg) # TMS 2
    sampe = cp.dot(OMS2,sampe)
    
    sampg += cp.random.normal(0, noise, size=(4,N))  # add noise (say of HEMT)
    sampe += cp.random.normal(0, noise, size=(4,N)) 
    
    ######
    sampg = cp.asnumpy(sampg)
    sampe = cp.asnumpy(sampe)
    
    hist_range = np.max(sampg[0:2,:])
    
    histg, x, y = np.histogram2d(sampg[0,:],sampg[1,:],range=[[-hist_range, hist_range],[-hist_range, hist_range]],bins=100)
    histe, x, y = np.histogram2d(sampe[0,:],sampe[1,:],range=[[-hist_range, hist_range],[-hist_range, hist_range]],bins=100)
    
    PDFg = histg/np.sum(histg) # create normalized probability density functions
    PDFe = histe/np.sum(histe)
    
    if plot:
    
        plt.figure()
        plt.pcolor(x,y,np.log(PDFg).transpose())
        plt.figure()
        plt.pcolor(x,y,np.log(PDFe).transpose())
        plt.figure()
        plt.pcolor(x,y,np.log(np.minimum(PDFg,PDFe).transpose()))
    
    fidelity =  1 - np.sum(np.minimum(PDFg,PDFe))/2 
    
    return fidelity, PDFg, PDFe, x, y


def TMS_exp(N, nbar=1, theta=np.pi, noise=0, r1=0, r2=0, phi1=0, phi2=0, plot=True):
    
    sigma = 0.4
    
    # define ops
    signal_rotation_g = cp.array(make_signalrotation_op(theta/2))
    signal_rotation_e = cp.array(make_signalrotation_op(-theta/2))
    TMS1 = cp.array(make_TMS_op(r1, phi1))
    TMS2 = cp.array(make_TMS_op(r2, phi2))
    
    sampg = cp.random.normal(0, sigma, size=(4,N)) # create random samples
    sampe = cp.random.normal(0, sigma, size=(4,N))
    
    sampg += cp.expand_dims(cp.array([nbar,0,0,0]),1) # displacement
    sampe += cp.expand_dims(cp.array([nbar,0,0,0]),1)
    
    sampg = cp.dot(TMS1,sampg) # TMS 1
    sampe = cp.dot(TMS1,sampe)
    
    sampg = cp.dot(signal_rotation_g,sampg) # dispersive shift
    sampe = cp.dot(signal_rotation_e,sampe)
    
    sampg = cp.dot(TMS2,sampg) # TMS 2
    sampe = cp.dot(TMS2,sampe)
    
    sampg += cp.random.normal(0, noise, size=(4,N))  # add noise (say of HEMT)
    sampe += cp.random.normal(0, noise, size=(4,N)) 
    
    ######
    sampg = cp.asnumpy(sampg)
    sampe = cp.asnumpy(sampe)
    
    hist_range = np.max([np.max(np.abs(sampg)),np.max(np.abs(sampe))])
    
    histg, x, y = np.histogram2d(sampg[0,:],sampg[1,:],range=[[-hist_range, hist_range],[-hist_range, hist_range]],bins=100)
    histe, x, y = np.histogram2d(sampe[0,:],sampe[1,:],range=[[-hist_range, hist_range],[-hist_range, hist_range]],bins=100)
    
    PDFg = histg/np.sum(histg) # create normalized probability density functions
    PDFe = histe/np.sum(histe)
    
    if plot:
    
        plt.figure()
        plt.pcolor(x,y,np.log(PDFg).transpose())
        plt.figure()
        plt.pcolor(x,y,np.log(PDFe).transpose())
        plt.figure()
        plt.pcolor(x,y,np.log(np.minimum(PDFg,PDFe).transpose()))
    
    fidelity =  1 - np.sum(np.minimum(PDFg,PDFe))/2 
    
    xc = (x[1:]+x[0:-1])/2
    yc = (y[1:]+y[0:-1])/2
    
    return fidelity, PDFg, PDFe, xc, yc
    
def TMSalt_exp(N, nbar=1, theta=np.pi, noise=0, r1=0, r2=0, phi1=0, phi2=0, plot=True):
    
    '''
    case where you drive the cavity through the qubit port
    '''
    
    sigma = 0.4
    
    # define ops
    signal_rotation_g = cp.array(make_signalrotation_op(theta/2))
    signal_rotation_e = cp.array(make_signalrotation_op(-theta/2))
    TMS1 = cp.array(make_TMS_op(r1, phi1))
    TMS2 = cp.array(make_TMS_op(r2, phi2))
    
    sampg = cp.random.normal(0, sigma, size=(4,N)) # create random samples
    sampe = cp.random.normal(0, sigma, size=(4,N))
    
    sampg = cp.dot(TMS1,sampg) # TMS 1
    sampe = cp.dot(TMS1,sampe)
    
    sampg += cp.expand_dims(cp.array([nbar,0,0,0]),1) # displacement
    sampe += cp.expand_dims(cp.array([nbar,0,0,0]),1)
    
    sampg = cp.dot(signal_rotation_g,sampg) # dispersive shift
    sampe = cp.dot(signal_rotation_e,sampe)
    
    sampg = cp.dot(TMS2,sampg) # TMS 2
    sampe = cp.dot(TMS2,sampe)
    
    sampg += cp.random.normal(0, noise, size=(4,N))  # add noise (say of HEMT)
    sampe += cp.random.normal(0, noise, size=(4,N)) 
    
    ######
    sampg = cp.asnumpy(sampg)
    sampe = cp.asnumpy(sampe)
    
    hist_range = np.max([np.max(np.abs(sampg)),np.max(np.abs(sampe))])
    
    histg, x, y = np.histogram2d(sampg[0,:],sampg[1,:],range=[[-hist_range, hist_range],[-hist_range, hist_range]],bins=100)
    histe, x, y = np.histogram2d(sampe[0,:],sampe[1,:],range=[[-hist_range, hist_range],[-hist_range, hist_range]],bins=100)
    
    PDFg = histg/np.sum(histg) # create normalized probability density functions
    PDFe = histe/np.sum(histe)
    
    if plot:
    
        plt.figure()
        plt.pcolor(x,y,np.log(PDFg).transpose())
        plt.figure()
        plt.pcolor(x,y,np.log(PDFe).transpose())
        plt.figure()
        plt.pcolor(x,y,np.log(np.minimum(PDFg,PDFe).transpose()))
    
    fidelity =  1 - np.sum(np.minimum(PDFg,PDFe))/2 
    
    xc = (x[1:]+x[0:-1])/2
    yc = (y[1:]+y[0:-1])/2
    
    return fidelity, PDFg, PDFe, xc, yc
    

#%% Gain test


r2s = np.linspace(0,5,5)
fidelities = r2s*0

for i in range(0,len(r2s)):
    
    
    fidelities[i], PDFg, PDFe, x, y = TMS_exp(1000000, nbar=1, theta=np.pi, noise=0, r1=0, r2=r2s[i], phi1=0, phi2=0, plot=False)
    
    plt.figure()
    plt.pcolor(x,y,(PDFg-PDFe).transpose(), cmap='seismic')
    
plt.figure()
plt.plot(r2s, fidelities)
plt.xlabel('Following amp r parameter')
plt.ylabel('Fidelity')


#%% TMS relative phase sweep

phi2s = np.linspace(0, 2*np.pi, 9)
fidelities = phi2s*0

xgavgs = fidelities*0
ygavgs = fidelities*0
xeavgs = fidelities*0
yeavgs = fidelities*0

varg = fidelities*0
vare = fidelities*0


for i in range(0,len(phi2s)):
    
    fidelities[i], PDFg, PDFe, x, y = TMS_exp(1000000, nbar=0, theta=np.pi/4, noise=1, r1=1, r2=2, phi1=0, phi2=phi2s[i], plot=False)
    
    xx,yy=np.meshgrid(x,y,indexing='ij')
    
    xgavgs[i] = np.sum(PDFg*xx)
    ygavgs[i] = np.sum(PDFg*yy)
    xeavgs[i] = np.sum(PDFe*xx)
    yeavgs[i] = np.sum(PDFe*yy)
    
    varg[i] = np.sum(np.sqrt((PDFg*(xx-xgavgs[i]))**2 + (PDFg*(yy-ygavgs[i]))**2))
    vare[i] = np.sum(np.sqrt((PDFe*(xx-xeavgs[i]))**2 + (PDFe*(yy-yeavgs[i]))**2))
    
    limit = np.max([np.max(np.abs(PDFg)),np.max(np.abs(PDFg))])
    
    plt.figure()
    plt.pcolor(x,y,(PDFe-PDFg).transpose(), cmap='seismic',vmin=-limit,vmax=limit)
    
plt.figure()
plt.plot(phi2s, fidelities)
plt.xlabel('Following amp r parameter')
plt.ylabel('Fidelity')

plt.figure()
plt.plot(xgavgs, ygavgs,label='g')
plt.plot(xeavgs, yeavgs,label='e')
plt.legend()

plt.figure()
plt.plot(phi2s, varg,label='g')
plt.plot(phi2s, vare,label='e')
plt.legend()





#%%

fidelity, PDFg, PDFe, x, y = TMS_exp(1000000, nbar=3, theta=np.pi/4, noise=0, r1=2, r2=2, phi1=0, phi2=np.pi/3, plot=True)

