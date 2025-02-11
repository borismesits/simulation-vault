import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter
import cupy as cp
from sim import MCSqueezer
from plotter import plot_ge_sample_list
import time
from scipy.special import erfcinv


def OMS_exp(N, nbar=1, theta=np.pi, temp=0, loss=0.5, r1=0, r2=0, phi1=0, phi2=0, use_cp=False):

    sim = MCSqueezer(N, sigma=temp, use_cp=use_cp)

    sim.displace(nbar, 0, 0, 0)
    sim.one_mode_squeeze(phi1, r1)
    sim.conditionally_rotate_signal(theta)
    sim.hotloss(loss**2, temp)
    sim.one_mode_squeeze(phi2, r2)
    # sim.hotloss(0.9, noise)

    return sim

def OMS_exp_lossy(N, nbar=1, theta=np.pi, noise=0, r1=0, r2=0, phi1=0, phi2=0, use_cp=False):

    sim = MCSqueezer(N, sigma=nbar/np.sqrt(nbar), use_cp=use_cp)

    sim.displace(nbar, 0, 0, 0)
    sim.one_mode_squeeze(phi1, r1)
    sim.hotloss(0.9, noise)
    sim.conditionally_rotate_signal(theta)
    sim.hotloss(0.9, noise)
    sim.one_mode_squeeze(phi2, r2)
    sim.hotloss(0.9, noise)

    return sim

def xmsmt_exp_lossy(N, nbar=1, theta=np.pi, noise=0, r1=0, r2=0, phi1=0, phi2=0, use_cp=False):

    sim = MCSqueezer(N, sigma=nbar/np.sqrt(nbar), use_cp=use_cp)

    sim.one_mode_squeeze(phi1, r1)
    sim.hotloss(0.9, noise)
    sim.conditionally_displace_signal(np.pi/2, nbar)
    sim.hotloss(0.9, noise)
    sim.one_mode_squeeze(phi2, r2)
    sim.hotloss(0.9, noise)

    return sim


def TMS_exp(N, nbar=1, theta=np.pi, temp=0, HEMTtemp=20, loss=0, r1=0, r2=0, phi1=0, phi2=0, use_cp=False):

    sim = MCSqueezer(N, sigma=np.sqrt(temp), use_cp=use_cp)

    sim.displace(np.sqrt(nbar), 0, 0, 0)
    sim.two_mode_squeeze(phi1, r1)
    sim.hotloss((1 - loss) ** 2, np.sqrt(temp))
    sim.conditionally_rotate_signal(theta)
    sim.hotloss((1-loss)**2, np.sqrt(temp))
    sim.two_mode_squeeze(phi2, r2)
    # sim.addnoise(np.sqrt(HEMTtemp))

    return sim


def RC_exp(N, nbar=1, theta=np.pi, noise=0, r1=0, r2=0, phi1=0, phi2=0, kerr=1e3, use_cp=False):

    sim = MCSqueezer(N, sigma=1, use_cp=use_cp)
    # sim.displace(nbar, 0, 0, 0)
    sim.two_mode_squeeze(phi1, r1)
    sim.conditionally_rotate_signal(theta)
    sim.four_wave_selfkerr(kerr, 1e-7, 1001)
    sim.two_mode_squeeze(phi2, r2)
    sim.four_wave_selfkerr(kerr, 1e-7, 1001)

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


def TMSalt_exp(N, nbar=1, theta=np.pi, temp=0, HEMTtemp=20, loss=0, r1=0, r2=0, phi1=0, phi2=0, use_cp=False):

    sim = MCSqueezer(N, sigma=np.sqrt(temp), use_cp=use_cp)


    sim.two_mode_squeeze(phi1, r1)
    sim.hotloss((1 - loss) ** 2, np.sqrt(temp))
    sim.displace(0, np.sqrt(nbar),0, 0)
    sim.conditionally_rotate_signal(theta)
    sim.displace(np.sqrt(nbar), 0, 0, 0)
    sim.hotloss((1-loss)**2, np.sqrt(temp))
    sim.two_mode_squeeze(phi2, r2)
    # sim.addnoise(np.sqrt(HEMTtemp))

    return sim


def split_1MS(N, nbar=1, theta=np.pi, noise=0, r=0, phi_s=0, theta_b=0, phi_b=0, use_cp=False):
    '''
    what happens when you apply a beamsplitter to 1MS light?
    '''

    sim = MCSqueezer(N, sigma=nbar / np.sqrt(nbar), use_cp=use_cp)

    sim.one_mode_squeeze_signal(phi_s, r)
    # sim.displace(10, 0, 0, 0)
    sim.beamsplit_signal_idler(theta_b, phi_b)
    # sim.displace(nbar, 0, 0, 0)
    # sim.conditionally_rotate_signal(theta)
    # sim.two_mode_squeeze(phi2, r2)
    # sim.hotloss(0.9, noise)

    return sim

def TMS_sweep_phase(N, phi1s):

    for i in range(0, len(phi1s)):
        sim_complete = TMSalt_exp(N, nbar=30, theta=phi1s[i], noise=100, r1=0.0, r2=2, phi1=0, phi2=np.pi,
                               use_cp=True)

        hist_ranges = [100,100,100,100,200,200]

        plot_ge_sample_list(sim_complete, savename=r'E:/TMS_anims/TMS/PP_theta' + str(i).zfill(3) + '.png', hist_ranges=hist_ranges)

def barzanjeh_repro_FIG7A(N, thetas, loss=0, phi2=np.pi):

    SNR_SU11PA = np.zeros([len(thetas)])
    SNR_PA = np.zeros([len(thetas)])

    for i in range(0, len(thetas)):
        sim_complete = TMS_exp(N, nbar=5, theta=thetas[i], temp=0.5, r1=1, loss=loss, r2=3, phi1=0,
                               phi2=phi2, use_cp=False)
        fidelities = plot_ge_sample_list(sim_complete, savename=None, fidonly=True)

        stdg = np.std(
            sim_complete.samples_history[-1][0][0:N // 2] + 1j * sim_complete.samples_history[-1][1][0:N // 2])
        stde = np.std(
            sim_complete.samples_history[-1][0][N // 2:] + 1j * sim_complete.samples_history[-1][1][0:N // 2:])
        meang = np.mean(
            sim_complete.samples_history[-1][0][0:N // 2] + 1j * sim_complete.samples_history[-1][1][0:N // 2])
        meane = np.mean(
            sim_complete.samples_history[-1][0][N // 2:] + 1j * sim_complete.samples_history[-1][1][N // 2:])

        SNR_SU11PA[i] = np.abs(meang - meane) / ((stdg + stde) / 2)  # denominator not quite right, check later

        # SNR_SU11PA[i, j] = np.sqrt(2)*erfcinv(2-2*fidelities[-1, 0])

        sim_complete = TMS_exp(N, nbar=5, theta=thetas[i], temp=0.5, r1=0, r2=3, loss=loss, phi1=0, phi2=phi2,
                               use_cp=False)
        fidelities = plot_ge_sample_list(sim_complete, savename=None, fidonly=True)

        stdg = np.std(
            sim_complete.samples_history[-1][0][0:N // 2] + 1j * sim_complete.samples_history[-1][1][0:N // 2])
        stde = np.std(
            sim_complete.samples_history[-1][0][N // 2:] + 1j * sim_complete.samples_history[-1][1][0:N // 2:])
        meang = np.mean(
            sim_complete.samples_history[-1][0][0:N // 2] + 1j * sim_complete.samples_history[-1][1][0:N // 2])
        meane = np.mean(
            sim_complete.samples_history[-1][0][N // 2:] + 1j * sim_complete.samples_history[-1][1][N // 2:])

        SNR_PA[i] = np.abs(meang - meane) / ((stdg + stde) / 2)  # denominator not quite right, check later

        # SNR_PA[i, j] = np.sqrt(2) * erfcinv(2 - 2 * fidelities[-1, 0])

        print(i)

    return SNR_SU11PA, SNR_PA


def barzanjeh_repro_lossy(N, thetas, losses):

    SNR_SU11PA = np.zeros([len(thetas), len(losses)])
    SNR_PA = np.zeros([len(thetas), len(losses)])

    for i in range(0, len(thetas)):

        for j in range(0, len(losses)):
            sim_complete = TMS_exp(N, nbar=5, theta=thetas[i], temp=0.5, r1=1, loss=losses[j], r2=3, phi1=0, phi2=np.pi, use_cp=False)
            fidelities = plot_ge_sample_list(sim_complete, savename=None, fidonly=True)

            stdg = np.std(
                sim_complete.samples_history[-1][0][0:N // 2] + 1j * sim_complete.samples_history[-1][1][0:N // 2])
            stde = np.std(
                sim_complete.samples_history[-1][0][N // 2:] + 1j * sim_complete.samples_history[-1][1][0:N // 2:])
            meang = np.mean(
                sim_complete.samples_history[-1][0][0:N // 2] + 1j * sim_complete.samples_history[-1][1][0:N // 2])
            meane = np.mean(
                sim_complete.samples_history[-1][0][N // 2:] + 1j * sim_complete.samples_history[-1][1][N // 2:])

            SNR_SU11PA[i,j] = np.abs(meang - meane) / ((stdg + stde) / 2)  # denominator not quite right, check later

            # SNR_SU11PA[i, j] = np.sqrt(2)*erfcinv(2-2*fidelities[-1, 0])

            sim_complete = TMS_exp(N, nbar=5, theta=thetas[i], temp=0.5, r1=0, r2=3, loss=losses[j], phi1=0, phi2=np.pi, use_cp=False)
            fidelities = plot_ge_sample_list(sim_complete, savename=None, fidonly=True)


            stdg = np.std(
                sim_complete.samples_history[-1][0][0:N // 2] + 1j * sim_complete.samples_history[-1][1][0:N // 2])
            stde = np.std(
                sim_complete.samples_history[-1][0][N // 2:] + 1j * sim_complete.samples_history[-1][1][0:N // 2:])
            meang = np.mean(
                sim_complete.samples_history[-1][0][0:N // 2] + 1j * sim_complete.samples_history[-1][1][0:N // 2])
            meane = np.mean(
                sim_complete.samples_history[-1][0][N // 2:] + 1j * sim_complete.samples_history[-1][1][N // 2:])

            SNR_PA[i, j] = np.abs(meang - meane) / ((stdg + stde))  # denominator not quite right, check later

            # SNR_PA[i, j] = np.sqrt(2) * erfcinv(2 - 2 * fidelities[-1, 0])

        # fidelity_ratios[i] = fidelity_SU11PA/fidelity_PA
        print(i)

    return SNR_SU11PA, SNR_PA

if __name__ == '__main__':
    # sim_complete = TMS_exp(10000000, nbar=10, theta=np.pi/10, noise=10, r1=2, r2=3, phi1=0, phi2=np.pi*0.9, use_cp=True)

    # sim_complete = OMS_exp(1000000, nbar=10, theta=np.pi/10, noise=10, r1=1.5, r2=1.5, phi1=0, phi2=np.pi,use_cp=True)
    # sim_complete = xmsmt_exp_lossy(10000000, nbar=10, theta=np.pi / 10, noise=0, r1=1.5, r2=1.5, phi1=0, phi2=np.pi,
    #                              use_cp=True)
    # sim_complete = RC_exp(100000, nbar=10, theta=np.pi/2, noise=1, r1=2, r2=2, phi1=0, phi2=np.pi/2, kerr=3e3, use_cp=True)
    # sim_complete = split_1MS(100000, nbar=1, theta=np.pi, noise=0, r=1.5, phi_s=0, theta_b=np.pi/4, phi_b=0, use_cp=False)
    # sim_complete = MZinterferometer(100000, nbar=5, theta=np.pi, phi=np.pi, noise=0, use_cp=False)

    # chikapparatio = np.linspace(0, 5, 50)
    # thetas = 4*np.arctan(chikapparatio)
    # SNR_SU11PA, SNR_PA = barzanjeh_repro_lossy(100000, thetas)
    #
    # plt.figure(1)
    # plt.plot(chikapparatio, SNR_SU11PA, 'b')
    # plt.plot(chikapparatio, SNR_PA, 'r')
    # plt.grid()

    N = 10000000

    sim_complete = TMS_exp(N, nbar=5, theta=np.pi/10, temp=0.5, loss=0.0, r1=1, r2=3, phi1=0, phi2=np.pi, use_cp=False)
    plot_ge_sample_list(sim_complete, savename=None, bins=100)

    stdg = np.std(
        sim_complete.samples_history[-1][0][0:N // 2] + 1j * sim_complete.samples_history[-1][1][0:N // 2])
    stde = np.std(
        sim_complete.samples_history[-1][0][N // 2:] + 1j * sim_complete.samples_history[-1][1][0:N // 2:])
    meang = np.mean(
        sim_complete.samples_history[-1][0][0:N // 2] + 1j * sim_complete.samples_history[-1][1][0:N // 2])
    meane = np.mean(
        sim_complete.samples_history[-1][0][N // 2:] + 1j * sim_complete.samples_history[-1][1][N // 2:])

    SNR = np.abs(meang - meane) / ((stdg + stde) / 2)  # denominator not quite right, check later

    sim_complete = TMSalt_exp(N, nbar=0, theta=np.pi, temp=0.5, loss=0.5, r1=0, r2=0, phi1=0, phi2=np.pi, use_cp=False)
    plot_ge_sample_list(sim_complete, savename=None, bins=50)

    stdg = np.std(
        sim_complete.samples_history[-1][0][0:N // 2] + 1j * sim_complete.samples_history[-1][1][0:N // 2])
    stde = np.std(
        sim_complete.samples_history[-1][0][N // 2:] + 1j * sim_complete.samples_history[-1][1][0:N // 2:])
    meang = np.mean(
        sim_complete.samples_history[-1][0][0:N // 2] + 1j * sim_complete.samples_history[-1][1][0:N // 2])
    meane = np.mean(
        sim_complete.samples_history[-1][0][N // 2:] + 1j * sim_complete.samples_history[-1][1][N // 2:])

    SNR = np.abs(meang - meane) / ((stdg + stde) / 2)  # denominator not quite right, check later



