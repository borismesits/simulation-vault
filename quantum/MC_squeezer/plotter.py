import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

try:
    import cupy as cp
except:
    pass

def plot_ge(sampg, sampe, idx1, idx2, ax, hist_range, lib):
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

    # axs.pcolormesh(xc,yc,(PDFe-PDFg).transpose(), cmap='seismic',vmin=-limit,vmax=limit,shading='gouraud')

    total = (PDFg + PDFe) / np.max(PDFg + PDFe)
    img = (PDFe - PDFg) / (PDFe + PDFg + 1e-12)

    G = img * 0
    R = 0.5 - img / 4
    B = 0.5 + img / 2
    A = total * 2

    rgba = np.stack([R, G, B, A], axis=-1)
    # rgba = np.flip(rgba, axis=0)
    rgba = np.clip(rgba, a_min=0, a_max=1)

    # ax.set_xticks(np.arange(-int(hist_range/10)*10, int(hist_range/10)*10, 10), minor=True)
    # ax.set_yticks(np.arange(-int(hist_range/10)*10, int(hist_range/10)*10, 10), minor=True)

    # ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)

    ax.imshow(np.transpose(rgba, axes=(1, 0, 2)), extent=[-hist_range, hist_range, -hist_range, hist_range], interpolation='bicubic')

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


def plot_ge_sample_list(sim):

    num_steps = len(sim.samples_history)

    fig, axs = plt.subplots(nrows=3, ncols=num_steps)

    for i in range(0, num_steps):

        sampg = sim.samples_history[i][:, 0:sim.N//2]
        sampe = sim.samples_history[i][:, sim.N//2:]

        lib = sim.lib

        # hist_range = np.float64(lib.max(lib.abs(sim.samples_history[i]))) # take the maximum radius point as hist range
        hist_range = np.float64(5*lib.sort(lib.max(lib.abs(sim.samples_history[i]),axis=0))[sim.N//2]) # take the median

        plot_ge(sampg, sampe, 0, 1, axs[0, i], hist_range, sim.lib)
        axs[0, i].set_title(sim.step_names[i])
        if i == 0:
            axs[0, i].set_ylabel('Signal IQ')

        plot_ge(sampg, sampe, 2, 3, axs[1, i], hist_range, sim.lib)
        if i == 0:
            axs[1, i].set_ylabel('Idler IQ')
        # cross I

        plot_ge(sampg, sampe, 0, 2, axs[2, i], hist_range, sim.lib)
        if i == 0:
            axs[2, i].set_ylabel('Signal-Idler II')

    plt.show()
