import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter
import cupy as cp
from sim import MCSqueezer
from plotter import plot_ge_sample_list
import time
from scipy.special import erfcinv, erfc


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
    sim.addnoise(np.sqrt(HEMTtemp))

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
            sim_complete = TMS_exp(N, nbar=5, theta=thetas[i], temp=0.5, r1=1, r2=3, loss=losses[j], phi1=0, phi2=np.pi, use_cp=False)
            fidelities = plot_ge_sample_list(sim_complete, savename=None, fidonly=True)

            stdg = np.std(
                sim_complete.samples_history[-1][0][0:N // 2] + 1j * sim_complete.samples_history[-1][1][0:N // 2])
            stde = np.std(
                sim_complete.samples_history[-1][0][N // 2:] + 1j * sim_complete.samples_history[-1][1][0:N // 2:])
            meang = np.mean(
                sim_complete.samples_history[-1][0][0:N // 2] + 1j * sim_complete.samples_history[-1][1][0:N // 2])
            meane = np.mean(
                sim_complete.samples_history[-1][0][N // 2:] + 1j * sim_complete.samples_history[-1][1][N // 2:])

            SNR_SU11PA[i,j] = np.abs(meang - meane) / ((stdg + stde)/2)  # denominator not quite right, check later

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

            SNR_PA[i, j] = np.abs(meang - meane) / ((stdg + stde)/2)  # denominator not quite right, check later

            # SNR_PA[i, j] = np.sqrt(2) * erfcinv(2 - 2 * fidelities[-1, 0])

        # fidelity_ratios[i] = fidelity_SU11PA/fidelity_PA
        print(i)

    return SNR_SU11PA, SNR_PA

if __name__ == '__main__':

    chikapparatio = np.linspace(0, 2, 50)
    thetas = 4*np.arctan(chikapparatio)
    losses = 10**np.linspace(-3, -0.5, 15)
    SNR_SU11PA, SNR_PA = barzanjeh_repro_lossy(100000, thetas, losses=losses)

    IF_SU11PA = 0.5*erfc(SNR_SU11PA/np.sqrt(2))
    IF_PA = 0.5 * erfc(SNR_PA / np.sqrt(2))

    fig, ax = plt.subplots()
    CS = ax.contour(np.log10(losses), chikapparatio, np.log10(IF_SU11PA))
    plt.grid()
    ax.clabel(CS, CS.levels)
    ax.set_xlabel('Log10 Circulator Loss')
    ax.set_ylabel('$2 \kappa / \chi$')
    ax.set_title('Log10(1-fidelity), SU11+PA')

    fig, ax = plt.subplots()
    CS = ax.contour(np.log10(losses), chikapparatio, np.log10(IF_PA))
    plt.grid()
    ax.clabel(CS, CS.levels)
    ax.set_xlabel('Log10 Circulator Loss')
    ax.set_ylabel('$2 \kappa / \chi$')
    ax.set_title('Log10 (1-fidelity), PA')

    fig, ax = plt.subplots()
    ax.plot(chikapparatio, SNR_SU11PA[:,0],'b',label='SU11+PA')
    ax.plot(chikapparatio, SNR_PA[:,0],'r', label='PA')

    theta = 2 * np.arctan(chikapparatio)
    G = 100
    AN = 0.5 * (1 - 1 / G)
    nbar = 5
    SNR_PA_th = 2 * np.sqrt(nbar) * np.abs(np.sin(theta)) / np.sqrt(2 * AN + 1)
    G1 = 2
    G2 = 100
    AN1 = 0.5 * (1 - 1 / G1)
    AN2 = 0.5 * (1 - 1 / G2)
    SNR_SU11PA_th = 2 * np.sqrt(nbar) * np.abs(np.sin(theta)) / np.sqrt(
        (2 * AN1 + 1) * (2 * AN2 + 1) - 8 * np.cos(theta) * np.sqrt(AN1 * AN2))
    ax.plot(chikapparatio, SNR_SU11PA_th, 'b--', label='Barzanjeh et al. Eq. 11')
    ax.plot(chikapparatio, SNR_PA_th, 'r--', label='Barzanjeh et al. Eq. 9')
    plt.grid()
    ax.set_ylabel('SNR')
    ax.set_xlabel('$2 \kappa / \chi$')
    ax.legend()


    def fmt(x):
        s = f"{x:.1f}"
        if s.endswith("0"):
            s = f"{x:.0f}"
        return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"

    fig, ax = plt.subplots(figsize=(5,4), dpi=80)
    # ax.pcolor(10*np.log10(losses), chikapparatio, 100 * (1 - IF_SU11PA))
    CS = ax.contour(10 * np.log10(1 - losses), chikapparatio, 100 * (1 - IF_SU11PA), levels=[90, 97, 99, 99.7],
                    colors=(0,0,0,0.2))
    CS = ax.contour(10*np.log10(1-losses), chikapparatio, 100*(1-IF_SU11PA), levels=[90, 97, 99, 99.7], colors='k')
    plt.grid()
    ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=8)
    ax.set_xlabel('Insertion Loss (dB)')
    ax.set_ylabel('$2 \kappa / \chi$')
    ax.set_title('Fidelity, SU11+PA')
    fig.tight_layout()

    # compare interferometer to PA
    from matplotlib.ticker import FormatStrFormatter
    fig, ax = plt.subplots(figsize=(4,3), dpi=200)
    PC = ax.pcolormesh(np.log10(losses), chikapparatio, IF_PA/IF_SU11PA,shading='gouraud',cmap='coolwarm')
    plt.grid()
    ax.set_xlabel('Loss')
    ax.set_ylabel('$2 \kappa / \chi$')
    ax.set_title('Infidelity PA / Infidelity SU11+PA ')
    labels = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
    ax.set_xticks([-3, -2.5, -2, -1.5, -1, -0.5], ['0.1%','0.3%', '1%', '3%', '10%', '30%'])
    fig.colorbar(PC, ax=ax)
    fig.tight_layout()



    # chikapparatio = np.linspace(0, 5, 50)
    # thetas = 4 * np.arctan(chikapparatio)
    # SNR_SU11PA, SNR_PA = barzanjeh_repro_FIG7A(100000, thetas, phi2=0)
    #
    # plt.figure(1)
    # plt.plot(chikapparatio, SNR_SU11PA / SNR_PA)
    # plt.grid()

    # sim_complete = OMS_exp(10000000, nbar=1, theta=np.pi/8, temp=0.5, loss=0.8, r1=1, r2=1, phi1=0, phi2=np.pi, use_cp=False)
    # plot_ge_sample_list(sim_complete, savename=None)

    # N = 1000000
    #
    # sim_complete = TMS_exp(N, nbar=3, theta=np.pi, temp=0.5, loss=0.0, r1=0, r2=1, phi1=0, phi2=np.pi, use_cp=False)
    # plot_ge_sample_list(sim_complete, savename=None)
    #
    # stdg = np.std(
    #     sim_complete.samples_history[-1][0][0:N // 2] + 1j * sim_complete.samples_history[-1][1][0:N // 2])
    # stde = np.std(
    #     sim_complete.samples_history[-1][0][N // 2:] + 1j * sim_complete.samples_history[-1][1][0:N // 2:])
    # meang = np.mean(
    #     sim_complete.samples_history[-1][0][0:N // 2] + 1j * sim_complete.samples_history[-1][1][0:N // 2])
    # meane = np.mean(
    #     sim_complete.samples_history[-1][0][N // 2:] + 1j * sim_complete.samples_history[-1][1][N // 2:])
    #
    # SNR = np.abs(meang - meane) / ((stdg + stde)/2)  # denominator not quite right, check later



