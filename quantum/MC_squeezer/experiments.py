import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter
import cupy as cp
from sim import MCSqueezer
from plotter import plot_ge_sample_list
import time

def OMS_exp(N, nbar=1, theta=np.pi, noise=0, r1=0, r2=0, phi1=0, phi2=0, use_cp=False):

    sim = MCSqueezer(N, sigma=nbar/np.sqrt(nbar), use_cp=use_cp)

    sim.displace(0, 0, 0, 0)
    sim.one_mode_squeeze(phi1, r1)
    # sim.conditionally_rotate_signal(theta)
    sim.one_mode_squeeze(phi2, r2)
    # sim.four_wave_selfkerr(10000, 1e-7, 1001)
    sim.hotloss(0.9, noise)

    return sim


def TMS_exp(N, nbar=1, theta=np.pi, noise=0, r1=0, r2=0, phi1=0, phi2=0, use_cp=False):

    sim = MCSqueezer(N, sigma=np.sqrt(nbar)+0.5, use_cp=use_cp)

    sim.displace(nbar, 0, 0, 0)
    sim.two_mode_squeeze(phi1, r1)
    sim.conditionally_rotate_signal(theta)
    sim.two_mode_squeeze(phi2, r2)
    sim.hotloss(0.9, noise)

    return sim


def RC_exp(N, nbar=1, theta=np.pi, noise=0, r1=0, r2=0, phi1=0, phi2=0, kerr=1e3, use_cp=False):

    sim = MCSqueezer(N, sigma=1, use_cp=use_cp)
    # sim.displace(nbar, 0, 0, 0)
    sim.one_mode_squeeze(phi1, r1)
    sim.conditionally_rotate_signal(theta)
    sim.four_wave_selfkerr(1e5, 1e-7, 1001)
    sim.one_mode_squeeze(phi2, r2)

    sim.hotloss(0.9, noise)

    return sim

def MZinterferometer(N, nbar=1, theta=np.pi, phi=0, noise=0, use_cp=False):

    sim = MCSqueezer(N, sigma=1, use_cp=use_cp)
    sim.displace(nbar, 0, 0, 0)
    sim.beamsplit_signal_idler(phi)
    sim.conditionally_rotate_signal(theta)
    sim.beamsplit_signal_idler(phi)
    sim.hotloss(0.9, noise)

    return sim


def PPamp_exp(N, nbar=1, theta=np.pi, noise=0, r=0, phi=0, use_cp=False):

    sim = MCSqueezer(N, sigma=nbar / np.sqrt(nbar), use_cp=use_cp)

    sim.displace(nbar, 0, 0, 0)
    sim.conditionally_rotate_signal(theta)
    sim.two_mode_squeeze(phi, r)
    sim.hotloss(0.9, noise)

    return sim

def PSamp_exp(N, nbar=1, theta=np.pi, noise=0, r=0, phi=0, use_cp=False):

    sim = MCSqueezer(N, sigma=nbar / np.sqrt(nbar), use_cp=use_cp)

    sim.displace(nbar, 0, 0, 0)
    sim.conditionally_rotate_signal(theta)
    sim.one_mode_squeeze(phi, r)
    sim.hotloss(0.9, noise)

    return sim


def TMSalt_exp(N, nbar=1, theta=np.pi, noise=0, r1=0, r2=0, phi1=0, phi2=0, use_cp=False):
    '''
    case where you drive the cavity through the qubit port
    '''

    sim = MCSqueezer(N, sigma=nbar / np.sqrt(nbar), use_cp=use_cp)

    sim.two_mode_squeeze(phi1, r1)
    sim.displace(nbar, 0, 0, 0)
    sim.conditionally_rotate_signal(theta)
    sim.two_mode_squeeze(phi2, r2)
    sim.hotloss(0.9, noise)

    return sim


if __name__ == '__main__':

    # sim_complete = TMS_exp(10000000, nbar=0, theta=np.pi/10, noise=10, r1=2, r2=3, phi1=0, phi2=np.pi*0.9, use_cp=True)

    # sim_complete = OMS_exp(1000000, nbar=1, theta=0, noise=1, r1=1, r2=2, phi1=0, phi2=np.pi/2,use_cp=True)
    sim_complete = RC_exp(100000, nbar=10, theta=np.pi/2, noise=1, r1=1, r2=1, phi1=0, phi2=np.pi/2, kerr=5e3, use_cp=True)
    # sim_complete = MZinterferometer(100000, nbar=5, theta=np.pi, phi=np.pi, noise=0, use_cp=False)

    plot_ge_sample_list(sim_complete)