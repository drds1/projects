import astropy.convolution as apc
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
import numpy as np
import scipy.signal as ss

import prediction_functions.polyfit as vpf


# plot x and y using generalised poly fit
def reg_plot(xlist, ylist, lab='', xlabel='', ylabel='', fig_title='', nlag=5):
    plt.clf()
    ny = len(ylist)
    rcoef = [0.0] * (ny)
    r2 = [0.0] * (ny)

    # try:
    ndown = 5
    nalong = ny

    if lab == '':
        ax_title = [''] * ny
    else:
        ax_title = lab

    if ylabel == '':
        ylab = [''] * ny
    else:
        ylab = ylabel

    if xlabel == '':
        xlab = [''] * ny
    else:
        xlab = xlabel

    ngrid = 1000

    yglo = []
    yghi = []
    ygmed = []
    xgs = []
    ordersave = []
    lagmax = []
    lagsave = []
    ccfsave = []

    idx = 0
    for y in ylist:
        x = xlist[idx]

        # perform ccf\
        nx = len(x)

        x_normalise = (x - x.mean()) / x.std()
        y_normalise = (y - y.mean()) / y.std()
        ccf = ss.correlate(x_normalise, y_normalise) / nx

        nccf = np.shape(ccf)[0]
        neg = np.int(np.floor(nccf - 1) / 2)
        lag = np.zeros(nccf)
        idlag = np.arange(neg + 1)
        lag[:neg] = -1 * idlag[-1:0:-1]
        lag[neg + 1:] = idlag[1:]

        # identify lag that maximizes ccf only shift if abs(lag) < nlag maximum number of months
        ccf_sorted = np.argsort(ccf)
        lags_sorted = lag[ccf_sorted]
        idx_allowed = np.where((lags_sorted < nlag) & (lags_sorted > -nlag))[0]
        lagmax_now = np.int(lags_sorted[idx_allowed[-1]])

        # if (np.abs(lagmax_now) > nlag):
        # lagmax_now = 0
        # lagmax_now = 0

        print('best lag = ', lagmax_now)
        kernel = np.zeros(nlag * 2 + 1)
        kernel[nlag + lagmax_now] = 1
        #y = apc.convolve(y, kernel)

        lagmax.append(lagmax_now)
        lagsave.append(lag)
        ccfsave.append(ccf)

        xmin, xmax = np.min(x), np.max(x)
        xrange = (xmax - xmin) / 10
        xgrid = np.linspace(xmin - xrange, xmax + xrange, ngrid)
        xgs.append(xgrid)

        a = vpf.fit_search(x, y, maxorder=2, xgrid=xgrid)
        order_cisqred, order_aic, order_bic = a
        order = order_aic
        ordersave.append(order)
        yg_med, yg_lo, yg_hi, cov, cisq, cisq_red, bic, aic, rmd, r2o = \
            vpf.fit(x, y, order=order, xgrid=xgrid, confidence=0.32, nits=20000, figure_title='',verbose=True)
        # correlation coefficient
        rcoef[idx] = np.corrcoef(x, y)[0, 1]
        r2[idx] = r2o
        yglo.append(yg_lo)
        yghi.append(yg_hi)
        ygmed.append(yg_med)
        # make figure
        idx = idx + 1

    if (fig_title != ''):
        gs1 = gridspec.GridSpec(ndown, nalong)
        gs1.update(left=0.1, right=0.98, wspace=0.55, hspace=0.0, bottom=0.1, top=0.93)
        for i in range(ny):
            xgrid = xgs[i]
            xmin, xmax = np.min(xgrid), np.max(xgrid)
            y = ylist[i]
            x = xlist[i]
            yg_lo = yglo[i]
            yg_hi = yghi[i]
            ymin, ymax = np.min(y), np.max(y)
            yrange = (ymax - ymin) / 10
            ax1 = plt.subplot(gs1[0:3, i])
            ax1.scatter(x, y, color='k', s=4)
            ax1.plot(xgrid, yg_lo, color='b', label='fit')
            ax1.plot(xgrid, yg_hi, color='b', label=None)
            ax1.fill_between(xgrid, yg_lo, yg_hi, alpha=0.3, color='b', label=None)
            ax1.set_xlabel(xlab[i])
            ax1.set_ylabel(ylab[i])
            ax1.set_title(ax_title[i])
            ax1.set_xlim([xmin, xmax])
            ax1.set_ylim([ymin - yrange, ymax + yrange])
            ax1.annotate(r'$r_c = ' + np.str(np.round(rcoef[i], 2)) + '$', (0.97, 0.92), xycoords='axes fraction',
                         horizontalalignment='right', color='b')
            ax1.annotate(r'$r^2 = ' + np.str(np.round(r2[i], 2)) + '$', (0.99, 0.78), xycoords='axes fraction',
                         horizontalalignment='right', color='b')
            ax1.annotate('order = ' + np.str(np.int(ordersave[i])), (0.99, 0.85), xycoords='axes fraction',
                         horizontalalignment='right', color='b')

            # plot ccf figure only include first nlag time steps
            ax2 = plt.subplot(gs1[4:, i])
            lag = lagsave[i]
            ccf = ccfsave[i]
            nccf = len(lag)
            nl0 = np.int(np.floor((nccf - 1) / 2))
            ilo = max(0, nl0 - nlag)
            ihi = min(nccf - 1, nl0 + nlag + 1)
            ax2.bar(lag[ilo:ihi], ccf[ilo:ihi], color='k')
            ylim_ccf = list(ax2.get_ylim())
            lagnow = lagmax[i]
            if (lagnow < 0):
                clag = 'green'
                ax1.annotate('lead = ' + np.str(np.int(np.abs(lagmax[i]))), (0.01, 0.92), xycoords='axes fraction',
                             horizontalalignment='left', color=clag)
            else:
                clag = 'red'
                ax1.annotate('lag = ' + np.str(np.int(np.abs(lagmax[i]))), (0.01, 0.92), xycoords='axes fraction',
                             horizontalalignment='left', color=clag)
            ax2.plot([lagnow] * 2, ylim_ccf, color=clag, linewidth=3, alpha=0.55)
            ax2.plot([0.0] * 2, ylim_ccf, ls='--', color='k')
            ax2.set_xlim([lag[ilo], lag[ihi]])
            ax2.set_xlabel('lag (days)')
            ax2.set_ylabel(r'$r_c$')
            ax2.annotate(r'ARB leads', (0.01, 0.82), fontsize=8, color='green', xycoords='axes fraction',
                         horizontalalignment='left')
            ax2.annotate('ARB lags', (0.97, 0.82), fontsize=8, color='red', xycoords='axes fraction',
                         horizontalalignment='right')

            plt.savefig(fig_title)

    # except:
    # pass

    return (rcoef, lagmax, xgs, yglo, ygmed, yghi, r2)
