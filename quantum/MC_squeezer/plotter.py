import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

try:
    import cupy as cp
except:
    pass

def plot_ge(sampg, sampe, idx1, idx2, ax, hist_range, lib, fidonly=False, cutoff=0, bins=30):
    '''
    Inputs:
        samples
        idx1 and idx2 determine which projection of the 4D space you want. 0 is I of signal, 1 is Q of signal, 2 I of idler, 3 Q of idler
        x,y a
    '''

    bins = bins

    histg, x, y = lib.histogram2d(sampg[idx1, :], sampg[idx2, :],
                                 range=[[-hist_range, hist_range], [-hist_range, hist_range]], bins=bins)
    histe, x, y = lib.histogram2d(sampe[idx1, :], sampe[idx2, :],
                                 range=[[-hist_range, hist_range], [-hist_range, hist_range]], bins=bins)

    histg[np.where(histg < cutoff)] = 0
    histe[np.where(histe < cutoff)] = 0

    if lib == cp:
        histg = cp.asnumpy(histg)
        histe = cp.asnumpy(histe)

    xc = (x[1:] + x[0:-1]) / 2
    yc = (y[1:] + y[0:-1]) / 2

    PDFg = histg / np.sum(histg)  # create normalized probability density functions
    PDFe = histe / np.sum(histe)

    limit = np.max([np.max(PDFg), np.max(PDFg)])

    fidelity = 1 - np.sum(np.minimum(PDFg, PDFe)) / 2

    if fidonly == False:

        total = (PDFg + PDFe) / np.max(PDFg + PDFe)
        img = (PDFe - PDFg) / (PDFe + PDFg + 1e-12)

        G = img * 0
        R = 0.5 - img / 4
        B = 0.5 + img / 2
        A = total * 3

        rgba = np.stack([R, G, B, A], axis=-1)
        rgba = np.clip(rgba, a_min=0, a_max=1)

        ax.imshow(np.transpose(rgba, axes=(1, 0, 2)), extent=[-hist_range, hist_range, -hist_range, hist_range], interpolation='gaussian')

        gr = np.sqrt(np.sum(xc * np.sum(PDFg, axis=0)) ** 2 + np.sum(yc * np.sum(PDFg, axis=1)) ** 2)
        er = np.sqrt(np.sum(xc * np.sum(PDFe, axis=0)) ** 2 + np.sum(yc * np.sum(PDFe, axis=1)) ** 2)

        ax.text(0.05, 0.95, 'F = ' + str(np.round(fidelity, decimals=3)), horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes)

        ax.text(0.05, 0.87, 'Gr = ' + str(np.round(gr, decimals=3)), horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes)

        ax.text(0.05, 0.79, 'Er = ' + str(np.round(er, decimals=3)), horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes)

    return fidelity

    # ax.scatter([sampg[idx2,0]],[sampg[idx1,0]],color='b',marker='1')
    # ax.scatter([sampg[idx2,1]],[sampg[idx1,1]],color='b',marker='2')
    # ax.scatter([sampg[idx2,2]],[sampg[idx1,2]],color='b',marker='3')
    # ax.scatter([sampg[idx2,3]],[sampg[idx1,3]],color='b',marker='4')

    # ax.scatter([sampe[idx2,0]],[sampe[idx1,0]],color='r',marker='1')
    # ax.scatter([sampe[idx2,1]],[sampe[idx1,1]],color='r',marker='2')
    # ax.scatter([sampe[idx2,2]],[sampe[idx1,2]],color='r',marker='3')
    # ax.scatter([sampe[idx2,3]],[sampe[idx1,3]],color='r',marker='4')

def plot_ge_alt(sampg, sampe, idx1, idx2, ax, hist_range, lib):
    '''
    Inputs:
        samples
        idx1 and idx2 determine which projection of the 4D space you want. 0 is I of signal, 1 is Q of signal, 2 I of idler, 3 Q of idler
        x,y a
    '''

    bins = 100

    histg, x, y = lib.histogram2d(sampg[idx1, :], sampg[idx2, :],
                                 range=[[-hist_range, hist_range], [-hist_range, hist_range]], bins=bins)
    histe, x, y = lib.histogram2d(sampe[idx1, :], sampe[idx2, :],
                                 range=[[-hist_range, hist_range], [-hist_range, hist_range]], bins=bins)

    if lib == cp:
        histg = cp.asnumpy(histg)
        histe = cp.asnumpy(histe)

    xc = (x[1:] + x[0:-1]) / 2
    yc = (y[1:] + y[0:-1]) / 2

    PDFg = histg / np.sum(histg)  # create normalized probability density functions
    PDFe = histe / np.sum(histe)

    limit = np.max([np.max(PDFg), np.max(PDFg)])

    conditional = np.round((PDFg-PDFe)/(PDFg+PDFe)*10)

    ax.imshow(conditional, extent=[-hist_range, hist_range, -hist_range, hist_range], vmin=-1, vmax=1, interpolation='bicubic',cmap='seismic')
    ax.set_facecolor((0.9, 0.9, 0.9))
    fidelity = np.round(1 - np.sum(np.minimum(PDFg, PDFe)) / 2, decimals=3)
    ax.text(0.05, 0.95, 'F = ' + str(fidelity), horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes)

    # ax.scatter([sampg[idx2,0]],[sampg[idx1,0]],color='b',marker='1')
    # ax.scatter([sampg[idx2,1]],[sampg[idx1,1]],color='b',marker='2')
    # ax.scatter([sampg[idx2,2]],[sampg[idx1,2]],color='b',marker='3')
    # ax.scatter([sampg[idx2,3]],[sampg[idx1,3]],color='b',marker='4')

    # ax.scatter([sampe[idx2,0]],[sampe[idx1,0]],color='r',marker='1')
    # ax.scatter([sampe[idx2,1]],[sampe[idx1,1]],color='r',marker='2')
    # ax.scatter([sampe[idx2,2]],[sampe[idx1,2]],color='r',marker='3')
    # ax.scatter([sampe[idx2,3]],[sampe[idx1,3]],color='r',marker='4')


def plot_ge_sample_list(sim, savename=None, hist_ranges=None, fidonly=False, cutoff=0, bins=30):

    num_steps = len(sim.samples_history)

    axs = np.zeros([3, num_steps])

    if fidonly == False:
        fig, axs = plt.subplots(nrows=3, ncols=num_steps, figsize=(num_steps*2-1,6-1), dpi=100)

    fidelities = np.zeros([num_steps, 3])

    for i in range(0, num_steps):

        sampg = sim.samples_history[i][:, 0:sim.N//2]
        sampe = sim.samples_history[i][:, sim.N//2:]

        lib = sim.lib

        # hist_range = np.float64(lib.max(lib.abs(sim.samples_history[i]))) # take the maximum radius point as hist range
        if hist_ranges == None:
            hist_range = np.float64(3*lib.sort(lib.max(lib.abs(sim.samples_history[i]),axis=0))[sim.N//2]) # take the median
        else:
            hist_range = hist_ranges[i]

        fidelities[i, 0] = plot_ge(sampg, sampe, 0, 1, axs[0, i], hist_range, sim.lib, fidonly=fidonly, cutoff=cutoff, bins=bins)
        fidelities[i, 1] = plot_ge(sampg, sampe, 2, 3, axs[1, i], hist_range, sim.lib, fidonly=fidonly, cutoff=cutoff, bins=bins)
        fidelities[i, 2] = plot_ge(sampg, sampe, 0, 2, axs[2, i], hist_range, sim.lib, fidonly=fidonly, cutoff=cutoff, bins=bins)

        if fidonly == False:

            axs[0, i].set_title(sim.step_names[i])

            axs[0,i].plot([-hist_range,hist_range],[0,0], 'k', linewidth=0.5)
            axs[0, i].plot([0, 0], [-hist_range, hist_range], 'k', linewidth=0.5)

            if i == 0:
                axs[0, i].set_ylabel('Signal IQ')
            if i == 0:
                axs[1, i].set_ylabel('Idler IQ')
            # cross I
            if i == 0:
                axs[2, i].set_ylabel('Signal-Idler II')

            axs[0, i].plot([-hist_range, hist_range], [0, 0], 'k', linewidth=0.5)
            axs[0, i].plot([0, 0], [-hist_range, hist_range], 'k', linewidth=0.5)
            axs[1, i].plot([-hist_range, hist_range], [0, 0], 'k', linewidth=0.5)
            axs[1, i].plot([0, 0], [-hist_range, hist_range], 'k', linewidth=0.5)
            axs[2, i].plot([-hist_range, hist_range], [0, 0], 'k', linewidth=0.5)
            axs[2, i].plot([0, 0], [-hist_range, hist_range], 'k', linewidth=0.5)

    if fidonly == False:
        plt.tight_layout()
        plt.show()

        if savename != None:
            plt.savefig(savename)

    return fidelities