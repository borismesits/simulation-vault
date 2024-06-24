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
    
    sigma = np.sqrt(nbar)+0.5
    
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
    
    sigma = np.sqrt(nbar)+0.5
    
    # define ops
    signal_rotation_g = cp.array(make_signalrotation_op(theta/2))
    signal_rotation_e = cp.array(make_signalrotation_op(-theta/2))
    TMS1 = cp.array(make_TMS_op(r1, phi1))
    TMS2 = cp.array(make_TMS_op(r2, phi2))
    
    sampgs = []
    sampes = []
    
    sampg, sampe = gen_vac_noise(N, sigma, testpoints=True)
    sampgs.append(cp.asnumpy(sampg))
    sampes.append(cp.asnumpy(sampe))
    
    sampg += cp.expand_dims(cp.array([nbar,0,0,0]),1) # displacement
    sampe += cp.expand_dims(cp.array([nbar,0,0,0]),1)
    sampgs.append(cp.asnumpy(sampg))
    sampes.append(cp.asnumpy(sampe))
    
    sampg = cp.dot(TMS1,sampg) # TMS 1
    sampe = cp.dot(TMS1,sampe)
    sampgs.append(cp.asnumpy(sampg))
    sampes.append(cp.asnumpy(sampe))
    
    sampg = cp.dot(signal_rotation_g,sampg) # dispersive shift
    sampe = cp.dot(signal_rotation_e,sampe)
    sampgs.append(cp.asnumpy(sampg))
    sampes.append(cp.asnumpy(sampe))
    
    sampg = cp.dot(TMS2,sampg) # TMS 2
    sampe = cp.dot(TMS2,sampe)
    sampgs.append(cp.asnumpy(sampg))
    sampes.append(cp.asnumpy(sampe))
    
    sampg += cp.random.normal(0, noise, size=(4,N))  # add noise (say of HEMT)
    sampe += cp.random.normal(0, noise, size=(4,N))
    sampgs.append(cp.asnumpy(sampg))
    sampes.append(cp.asnumpy(sampe))
    
    ######
    sampg = cp.asnumpy(sampg)
    sampe = cp.asnumpy(sampe)
    
    hist_range = np.max([np.max(np.abs(sampg)),np.max(np.abs(sampe))])
    
    histg, x, y = np.histogram2d(sampg[0,:],sampg[1,:],range=[[-hist_range, hist_range],[-hist_range, hist_range]],bins=200)
    histe, x, y = np.histogram2d(sampe[0,:],sampe[1,:],range=[[-hist_range, hist_range],[-hist_range, hist_range]],bins=200)
    
    PDFg = histg/np.sum(histg) # create normalized probability density functions
    PDFe = histe/np.sum(histe)
    
    if plot:
        
        titles = ['Noise, sigma = ' + str(sigma),'Displacement, nbar = ' + str(nbar),'TMS 1','Dispersive Shift','TMS 2','HEMT Noise']
    
        plot_sample_list(sampgs, sampes, titles)
    
    fidelity =  1 - np.sum(np.minimum(PDFg,PDFe))/2 
    
    xc = (x[1:]+x[0:-1])/2
    yc = (y[1:]+y[0:-1])/2
    
    return fidelity, PDFg, PDFe, xc, yc


def PPamp_exp(N, nbar=1, theta=np.pi, noise=0, r=0, phi=0, plot=True):
    
    sigma = np.sqrt(nbar)+0.5
    
    # define ops
    signal_rotation_g = cp.array(make_signalrotation_op(theta/2))
    signal_rotation_e = cp.array(make_signalrotation_op(-theta/2))
    TMS = cp.array(make_TMS_op(r, phi))
    
    sampgs = []
    sampes = []
    
    sampg, sampe = gen_vac_noise(N, sigma, testpoints=True)
    sampgs.append(cp.asnumpy(sampg))
    sampes.append(cp.asnumpy(sampe))
    
    sampg += cp.expand_dims(cp.array([nbar,0,0,0]),1) # displacement
    sampe += cp.expand_dims(cp.array([nbar,0,0,0]),1)
    sampgs.append(cp.asnumpy(sampg))
    sampes.append(cp.asnumpy(sampe))

    
    sampg = cp.dot(signal_rotation_g,sampg) # dispersive shift
    sampe = cp.dot(signal_rotation_e,sampe)
    sampgs.append(cp.asnumpy(sampg))
    sampes.append(cp.asnumpy(sampe))
    
    sampg = cp.dot(TMS,sampg) # TMS
    sampe = cp.dot(TMS,sampe)
    sampgs.append(cp.asnumpy(sampg))
    sampes.append(cp.asnumpy(sampe))
    
    sampg += cp.random.normal(0, noise, size=(4,N))  # add noise (say of HEMT)
    sampe += cp.random.normal(0, noise, size=(4,N))
    sampgs.append(cp.asnumpy(sampg))
    sampes.append(cp.asnumpy(sampe))
    
    ######
    sampg = cp.asnumpy(sampg)
    sampe = cp.asnumpy(sampe)
    
    hist_range = np.max([np.max(np.abs(sampg)),np.max(np.abs(sampe))])
    
    histg, x, y = np.histogram2d(sampg[0,:],sampg[1,:],range=[[-hist_range, hist_range],[-hist_range, hist_range]],bins=200)
    histe, x, y = np.histogram2d(sampe[0,:],sampe[1,:],range=[[-hist_range, hist_range],[-hist_range, hist_range]],bins=200)
    
    PDFg = histg/np.sum(histg) # create normalized probability density functions
    PDFe = histe/np.sum(histe)
    
    if plot:
        
        titles = ['Noise, sigma = ' + str(sigma),'Displacement, nbar = ' + str(nbar),'Dispersive Shift','2MS','HEMT Noise']
    
        plot_sample_list(sampgs, sampes, titles)
    
    fidelity =  1 - np.sum(np.minimum(PDFg,PDFe))/2 
    
    xc = (x[1:]+x[0:-1])/2
    yc = (y[1:]+y[0:-1])/2
    
    return fidelity, PDFg, PDFe, xc, yc

def PSamp_exp(N, nbar=1, theta=np.pi, noise=0, r=0, phi=0, plot=True):
    
    sigma = np.sqrt(nbar)+0.5
    
    # define ops
    signal_rotation_g = cp.array(make_signalrotation_op(theta/2))
    signal_rotation_e = cp.array(make_signalrotation_op(-theta/2))
    OMS = cp.array(make_OMS_op(r, phi))
    
    sampgs = []
    sampes = []
    
    sampg, sampe = gen_vac_noise(N, sigma, testpoints=True)
    sampgs.append(cp.asnumpy(sampg))
    sampes.append(cp.asnumpy(sampe))
    
    sampg += cp.expand_dims(cp.array([nbar,0,0,0]),1) # displacement
    sampe += cp.expand_dims(cp.array([nbar,0,0,0]),1)
    sampgs.append(cp.asnumpy(sampg))
    sampes.append(cp.asnumpy(sampe))

    
    sampg = cp.dot(signal_rotation_g,sampg) # dispersive shift
    sampe = cp.dot(signal_rotation_e,sampe)
    sampgs.append(cp.asnumpy(sampg))
    sampes.append(cp.asnumpy(sampe))
    
    sampg = cp.dot(OMS,sampg) # TMS
    sampe = cp.dot(OMS,sampe)
    sampgs.append(cp.asnumpy(sampg))
    sampes.append(cp.asnumpy(sampe))
    
    sampg += cp.random.normal(0, noise, size=(4,N))  # add noise (say of HEMT)
    sampe += cp.random.normal(0, noise, size=(4,N))
    sampgs.append(cp.asnumpy(sampg))
    sampes.append(cp.asnumpy(sampe))
    
    ######
    sampg = cp.asnumpy(sampg)
    sampe = cp.asnumpy(sampe)
    
    hist_range = np.max([np.max(np.abs(sampg)),np.max(np.abs(sampe))])
    
    histg, x, y = np.histogram2d(sampg[0,:],sampg[1,:],range=[[-hist_range, hist_range],[-hist_range, hist_range]],bins=200)
    histe, x, y = np.histogram2d(sampe[0,:],sampe[1,:],range=[[-hist_range, hist_range],[-hist_range, hist_range]],bins=200)
    
    PDFg = histg/np.sum(histg) # create normalized probability density functions
    PDFe = histe/np.sum(histe)
    
    if plot:
        
        titles = ['Noise, sigma = ' + str(sigma),'Displacement, nbar = ' + str(nbar),'Dispersive Shift','1MS','HEMT Noise']
    
        plot_sample_list(sampgs, sampes, titles)
    
    fidelity =  1 - np.sum(np.minimum(PDFg,PDFe))/2 
    
    xc = (x[1:]+x[0:-1])/2
    yc = (y[1:]+y[0:-1])/2
    
    return fidelity, PDFg, PDFe, xc, yc

def plot_ge(sampg, sampe, idx1, idx2, ax, hist_range):
    
    '''
    Inputs:
        samples
        idx1 and idx2 determine which projection of the 4D space you want. 0 is I of signal, 1 is Q of signal, 2 I of idler, 3 Q of idler
        x,y a
    '''
    
    bins = 100
    
    histg, x, y = np.histogram2d(sampg[idx1,:],sampg[idx2,:],range=[[-hist_range, hist_range],[-hist_range, hist_range]],bins=bins)
    histe, x, y = np.histogram2d(sampe[idx1,:],sampe[idx2,:],range=[[-hist_range, hist_range],[-hist_range, hist_range]],bins=bins)

    xc = (x[1:]+x[0:-1])/2
    yc = (y[1:]+y[0:-1])/2
    
    PDFg = histg/np.sum(histg) # create normalized probability density functions
    PDFe = histe/np.sum(histe)
    
    limit = np.max([np.max(PDFg),np.max(PDFg)])
    
    # axs.pcolormesh(xc,yc,(PDFe-PDFg).transpose(), cmap='seismic',vmin=-limit,vmax=limit,shading='gouraud')
    
    total = (PDFg+PDFe)/np.max(PDFg+PDFe)
    img = (PDFe-PDFg)/(PDFe+PDFg+1e-12)
    

    G = img*0
    R = 0.5-img/4
    B = 0.5+img/2
    A = total*2
    
    rgba = np.stack([R,G,B,A],axis=-1)
    rgba = np.flip(rgba,axis=0)
    
    # ax.set_xticks(np.arange(-int(hist_range/10)*10, int(hist_range/10)*10, 10), minor=True)
    # ax.set_yticks(np.arange(-int(hist_range/10)*10, int(hist_range/10)*10, 10), minor=True)
    
    # ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)

    ax.imshow(rgba,extent=[-hist_range, hist_range,-hist_range, hist_range],interpolation='bicubic')
    
    fidelity =  np.round(1 - np.sum(np.minimum(PDFg,PDFe))/2,decimals=3)
    ax.text(0.05, 0.95, 'F = ' + str(fidelity), horizontalalignment='left', verticalalignment='top', transform = ax.transAxes)
    
    # ax.scatter([sampg[idx2,0]],[sampg[idx1,0]],color='b',marker='1')
    # ax.scatter([sampg[idx2,1]],[sampg[idx1,1]],color='b',marker='2')
    # ax.scatter([sampg[idx2,2]],[sampg[idx1,2]],color='b',marker='3')
    # ax.scatter([sampg[idx2,3]],[sampg[idx1,3]],color='b',marker='4')
    
    # ax.scatter([sampe[idx2,0]],[sampe[idx1,0]],color='r',marker='1')
    # ax.scatter([sampe[idx2,1]],[sampe[idx1,1]],color='r',marker='2')
    # ax.scatter([sampe[idx2,2]],[sampe[idx1,2]],color='r',marker='3')
    # ax.scatter([sampe[idx2,3]],[sampe[idx1,3]],color='r',marker='4')
    

def plot_sample_list(sampgs, sampes, titles):
    
    num_steps = len(sampgs)
    
    fig, axs = plt.subplots(nrows=3, ncols=num_steps)
    
    for i in range(0,num_steps):
        
        sampg = sampgs[i]
        sampe = sampes[i]
        
        hist_range = np.max([np.max(np.abs(sampg)),np.max(np.abs(sampe))])
        
        # signal
        
        plot_ge(sampg, sampe, 0, 1, axs[0,i], hist_range)
        axs[0,i].set_title(titles[i])
        if i == 0:
            axs[0,i].set_ylabel('Signal IQ')

        plot_ge(sampg, sampe, 2, 3, axs[1,i], hist_range)
        if i == 0:
            axs[1,i].set_ylabel('Idler IQ')
        # cross I 
        
        plot_ge(sampg, sampe, 0, 2, axs[2,i], hist_range)
        if i == 0:
            axs[2,i].set_ylabel('Signal-Idler II')
    
def gen_vac_noise(N, sigma, testpoints=False):
    
    sampg = cp.random.normal(0, sigma, size=(4,N)) # create random samples
    sampe = cp.random.normal(0, sigma, size=(4,N))
    
    if testpoints==True:
        
        sampg[:,0] = cp.array([sigma,0,0,0])
        sampg[:,1] = cp.array([0,sigma,0,0])
        sampg[:,2] = cp.array([0,0,sigma,0])
        sampg[:,3] = cp.array([0,0,0,sigma])
        
        sampe[:,0] = cp.array([sigma,0,0,0])
        sampe[:,1] = cp.array([0,sigma,0,0])
        sampe[:,2] = cp.array([0,0,sigma,0])
        sampe[:,3] = cp.array([0,0,0,sigma])
    
    return sampg, sampe
    

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
    
    sampg, sampe = gen_vac_noise(N, sigma, testpoints=True)
    
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
    
    histg, x, y = np.histogram2d(sampg[0,:],sampg[1,:],range=[[-hist_range, hist_range],[-hist_range, hist_range]],bins=200)
    histe, x, y = np.histogram2d(sampe[0,:],sampe[1,:],range=[[-hist_range, hist_range],[-hist_range, hist_range]],bins=200)
    
    PDFg = histg/np.sum(histg) # create normalized probability density functions
    PDFe = histe/np.sum(histe)
    
    if plot:
    
        plot_sample_list()
    
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
    
    fidelities[i], PDFg, PDFe, x, y = TMS_exp(1000000, nbar=2, theta=np.pi/4, noise=1, r1=1, r2=2, phi1=0, phi2=phi2s[i], plot=True)
    

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

fidelity, PDFg, PDFe, x, y = TMS_exp(1000000, nbar=100, theta=np.pi/8, noise=10, r1=2, r2=2, phi1=0, phi2=np.pi, plot=True)

#%%

fidelity, PDFg, PDFe, x, y = PPamp_exp(1000000, nbar=100, theta=np.pi/8, noise=10, r=2, phi=0, plot=True)


#%%

fidelity, PDFg, PDFe, x, y = PSamp_exp(1000000, nbar=100, theta=np.pi/8, noise=10, r=2, phi=np.pi, plot=True)

