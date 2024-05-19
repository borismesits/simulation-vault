import numpy as np
import matplotlib.pyplot as plt

def add_db(dbs):
    '''
    adding two signals using only db units
    '''
    
    
    lins = 10**(np.array(dbs,dtype=np.float64)/10)
    lin = np.sum(lins)
    
    db = 10*np.log10(lin)
    
    return db

def temp2noisepwr(temp, bandwidth):
    
    k_B = 1.38e-23
    
    return bandwidth*temp*k_B

def lin2dBm(lin):
    
    dBm = 10*np.log10(lin*1000)
    
    return dBm

def dBm2lin(dBm):
    
    lin = 0.001*10**(dBm/10)
    
    return lin

def amplifier(signal, noise, gain=20, temp=300, noise_temp=300, out_sat_pwr=0, bw=1e6):
    
    
    signal = np.clip(signal+gain,-1e99, out_sat_pwr)
                     
    noise = add_db([noise+gain, lin2dBm(temp2noisepwr(temp+noise_temp, bw))])
    
    return signal, noise

def attenuator(signal, noise, att=20, temp=300, bw=1e6):
    
    signal = signal-att
                     
    noise = add_db([noise-att, lin2dBm(temp2noisepwr(temp, bw))])
    
    return signal, noise


bandwidth = 1e6

signal = 0
noise = lin2dBm(temp2noisepwr(300*10, bandwidth))

SNR = []
SNR.append(dBm2lin(signal-noise))

signal, noise = attenuator(signal, noise, att=20+5, temp=4*2, bw=bandwidth)
SNR.append(dBm2lin(signal-noise))

signal, noise = attenuator(signal, noise, att=20+5, temp=0.1*2, bw=bandwidth)
SNR.append(dBm2lin(signal-noise))

signal, noise = attenuator(signal, noise, att=20+5, temp=0.02*2, bw=bandwidth)
SNR.append(dBm2lin(signal-noise))

signal, noise = attenuator(signal, noise, att=30+5, temp=0.02*2, bw=bandwidth)  
SNR.append(dBm2lin(signal-noise))

signal, noise = amplifier(signal, noise, gain=35, temp=4, noise_temp=1.5, out_sat_pwr=10, bw=bandwidth)
SNR.append(dBm2lin(signal-noise))

signal, noise = attenuator(signal, noise, att=5+5, temp=4, bw=bandwidth)
SNR.append(dBm2lin(signal-noise))

signal, noise = amplifier(signal, noise, gain=30, temp=300, noise_temp=150, out_sat_pwr=10, bw=bandwidth)
SNR.append(dBm2lin(signal-noise))

signal, noise = amplifier(signal, noise, gain=20, temp=300, noise_temp=150, out_sat_pwr=30, bw=bandwidth)
SNR.append(dBm2lin(signal-noise))


print(signal)
print(noise)
print(SNR)
