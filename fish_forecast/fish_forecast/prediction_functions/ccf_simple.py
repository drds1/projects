import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
import scipy.signal as ss


def ccf_simp(ts1_in, ts2_in, resolution=1.0, r_crit=0.8, laglim='auto'):
    """
    Compute the CCF between two time series (basic version of the more
    sophisticated ccf_frrss function below)

    (+ve lags indicate ts2 leads ts1)

    :param ts1_in: Primary input time series - numpy 2D array [times, values]
    :param ts2_in: Secondary input time series - numpy 2D array [times, values]
    :param resolution: grid resolution in days (default 1 day). Higher res slows down code
    decrease (bigger number) for a speedup. Too high may degrade results.
    :param r_crit: Points < r_crit*ccfpeak are not considered
    :param laglim: Limits of ccf analysis. Code decides if 'auto',
    else enter a tuple (lowerlim,upperlim)
    :return:
    tccf,       The lag grid at which the ccf is evaluated (numpy 1D array)
    ccf,        The the ccf at each point in the lag grid (numpy 1D array)
    laglo:      Lag of Lower ccf limit (where ccf < r_crit*ccf - left side of peak)
    lagpeak:    Lag of ccf peak
    laghi:      Lag of upper ccf limit (where ccf < r_crit*ccf - left side of peak)
    idlo:  Index of the lower ccf limit (where ccf < r_crit*ccf - left side of peak)
    idmax: Index of the ccf peak
    idhi:   Index of the upper ccf limit (where ccf < r_crit*ccf - right side of peak)
    """
    ts1 = np.array(ts1_in, dtype=float)
    ts2 = np.array(ts2_in, dtype=float)


    # only select overlapping periods to take part in time series analysis
    tmin = max(np.min(ts1[:, 0]), np.min(ts2[:, 0]))
    tmax = min(np.max(ts1[:, 0]), np.max(ts2[:, 0]))

    # interpolate onto regular time grid
    tg = np.arange(tmin, tmax + resolution, resolution)
    yg1 = np.interp(tg, ts1[:, 0], ts1[:, 1])
    yg2 = np.interp(tg, ts2[:, 0], ts2[:, 1])

    yg1_mean = np.mean(yg1)
    yg1_sd = np.std(yg1)
    yg1 = (yg1 - yg1_mean) / yg1_sd

    yg2_mean = np.mean(yg2)
    yg2_sd = np.std(yg2)
    yg2 = (yg2 - yg2_mean) / yg2_sd

    # compute ccf
    ccf = ss.correlate(yg1, yg2)
    nccf = np.shape(ccf)[0]
    tccf = (np.arange(nccf) - nccf / 2) * resolution

    if type(laglim) is tuple:
        idccf = np.where((tccf > laglim[0]) & (tccf < laglim[1]))[0]
        ccf = ccf[idccf]
        tccf = tccf[idccf]
        nccf = np.shape(ccf)[0]


    ccf_max = np.max(ccf)

    # Peterson, Edelson 2018 definition of ccf inclusion threshold (>0.8 ccfmax)
    ccf_crit = r_crit * ccf_max
    idmax = np.argmax(ccf)

    try:
        idhi = idmax + np.where(ccf[idmax:] < ccf_crit)[0][0]
    except:
        print('ccf upper lag limit goes off the end of the grid. Re run with higher upper laglim')
        idhi = nccf - 1
    try:
        idlo = np.where(ccf[:idmax] < ccf_crit)[0][-1]
    except:
        idlo = 0
        print('ccf lower lag limit goes off the lower end of the grid. Re run with lower lower laglim')

    lagpeak = tccf[idmax]
    laglo = tccf[idlo]
    laghi = tccf[idhi]

    print('ccf low, peak, hi ', laglo, lagpeak, laghi)
    return (tccf, ccf, laglo, lagpeak, laghi, idlo, idmax, idhi)


# frrss ccf sampler
def ccf_frrss(ts1, ts2, resolution=1.0, fraction_rss=0.8, nsim=500,
              centroid_frac=0.8, flux_randomisation=0):
    """
    Cross correlation code to detect lags between time series 1 (ts1) and 2.
    Extra-galactic-astronomy-inspired Flux-Randomisation-Random-Subset-Selection
    (FR_RSS) technique https://ui.adsabs.harvard.edu/#abs/arXiv:astro-ph%2F9802103

    Generates nsim time series by sub-sampling the input and performs cross correlation
    on each. This code returns the probability distribution of CCF peaks and means.

    (+ve lags indicate ts2 leads ts1)

    :param ts1: Primary input time series - numpy 2D array [times, values]
    :param ts2: Secondary input time series - numpy 2D array [times, values]
    :param resolution: grid resolution in days (default 1 day). Higher res slows down code
        decrease (bigger number) for a speedup. Too high may degrade results.
    :param fraction_rss: Fraction of points included in the sub-set time series'
    :param nsim: number of sub-set time series' to draw (default = 500 should be fine)
    :param centroid_frac: Only results > 0.8*peak ccf value will count towards the final distribution
    :param flux_randomisation: Fraction of points from parent light curve present in the sub set

    :return: The output of the ccf_frrss function (preferable to use peaks rather than cenrtroids)
    lag_ccf:The lags considered,
    ccf_save: The correlation function at each lag considered for each of the nsim simulations
    numpy 2D array [len(lag_ccf), nsim]
    lagpeak: The distribution of ccf peaks at each time bin - numpy 1D array [len(lag_ccf)]
    ccfpeak: The ccf peak at each time bin - numpy 1D array [len(lag_ccf)]
    lagcent: The distribution of ccf centroids at each time bin - numpy 1D array [len(lag_ccf)]

    """
    n1 = np.shape(ts1)[0]
    n2 = np.shape(ts2)[0]
    n1_subset = np.int(fraction_rss * n1)
    n2_subset = np.int(fraction_rss * n2)

    # only overlapping periods take part in ccf analysis
    tmin = max(np.min(ts1[:, 0]), np.min(ts2[:, 0]))
    tmax = min(np.max(ts1[:, 0]), np.max(ts2[:, 0]))
    tgrid = np.arange(tmin, tmax + resolution, resolution)

    lagpeak = []
    ccfpeak = []
    ccf_save = []
    lagcent = []
    for i in range(nsim):
        id1 = np.sort(np.random.choice(np.arange(n1), size=n1_subset, replace=True))
        id2 = np.sort(np.random.choice(np.arange(n2), size=n2_subset, replace=True))

        # Perturb the time series by its uncertainty if present
        if (flux_randomisation == 1):
            noise1 = np.random.randn(n1_subset) * ts1[id1, 2]
            noise2 = np.random.randn(n2_subset) * ts2[id2, 2]
        else:
            noise1 = np.zeros(n1_subset)
            noise2 = np.zeros(n2_subset)

        # interpolate and standardize the time series, then cross correlating
        x1 = np.interp(tgrid, ts1[id1, 0], ts1[id1, 1] + noise1)
        x2 = np.interp(tgrid, ts2[id2, 0], ts2[id2, 1] + noise2)
        x1_new = (x1 - x1.mean()) / x1.std()
        x2_new = (x2 - x2.mean()) / x2.std()
        ccf = np.correlate(x1_new, x2_new, mode='full') / np.shape(x1_new)[0]

        if (i == 0):
            nccf = np.shape(ccf)[0]
            tccf = (np.arange(nccf) - np.floor(nccf / 2)) * resolution
            tmax = tccf[-1]
            tmin = tccf[0]
            lolim = tmin / 2
            hilim = tmax / 2
            idinc = np.where((tccf < hilim) & (tccf > lolim))[0]
            lag_ccf = tccf[idinc]

        idpeak = np.argmax(ccf[idinc])
        ccfp = ccf[idinc][idpeak]
        tpeak = tccf[idinc][idpeak]
        ccf_save.append(ccf[idinc])
        lagpeak.append(tpeak)
        ccfpeak.append(ccfp)
        ccfinc = ccf[idinc]
        laginc = tccf[idinc]
        idcent = np.where(ccfinc > centroid_frac * ccfp)[0]
        lagcent.append(np.sum(laginc[idcent] * ccfinc[idcent]) / np.sum(ccfinc[idcent]))

    return (np.array(lag_ccf), ccf_save, np.array(lagpeak), np.array(ccfpeak), np.array(lagcent))


def lag_analysis_figure(ts1, ts2, lagbins, ccfdistribution, ccf_stat,
                        xlabel = 'Time (days)',
                        titles=('time series 1', 'time series 2'),
                        dates_overide = None,
                        titles_ccf_stat='median ccf',
                        figure_title='show',
                        global_title='lag analysis',
                        laglims = (-60,60),
                        invert_colours=True):
    """
    Visualise the results of the fr-rss lag detection
    :param titles:
    :param ts1: Input time series (numpy 2d array [times,y])
    :param ts2: Second input time series (+ve lags indicate ts2 leads ts1)
    :param lagbins: The lag bins from the frrss ccf analysis (above)
    :param dates_overide: If not None then replace the x axis with this array
        useful for replacing normal time axis with date-time pandas dates
    :param ccfdistribution: The distribution value at each lag bin above
    :param ccf_stat: For each time bin, return i.e the max-mean-median ccf in the bin
    :param figure_title: Set to 'show' to display to screen or xxxxx.pdf etc to save
    :return: Figure plotted either to screen or saved to file (see 'figure_title' argument)
    """

    if invert_colours is True:
        c = 'cyan'
    else:
        c = 'purple'



    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    if dates_overide is not None:
        x = dates_overide
        ax1.tick_params(axis='x', rotation=45)
    else:
        x = ts1[:,0]
    ax1.plot(x, ts1[:, 1], color='b', label=titles[0])
    ax1.set_xlabel(xlabel)
    plt.setp(ax1.get_yticklabels(),color='b')
    ax1.tick_params(axis='y',color='b')
    ax1.set_ylabel(titles[0],color='b')
    ax1.legend(loc=1)
    ax2 = ax1.twinx()
    if dates_overide is not None:
        x = dates_overide
        ax2.tick_params(axis='x', rotation=45)
    else:
        x = ts2[:,0]
    ax2.plot(x, ts2[:, 1], color='r', label=titles[1])
    ax2.set_ylabel(titles[1],color='r')
    ax2.tick_params(axis='y', color='r')
    plt.setp(ax2.get_yticklabels(), color='r')
    ax2.legend(loc=0)
    ax1.set_title(global_title)

    ax3 = fig.add_subplot(212)
    ax3.hist(ccfdistribution, bins=lagbins, color=c, alpha=0.4)
    ax3.set_xlim(laglims)
    ax3.set_xlabel('lag (days)'+'\n '+titles[0]+' leads <------------------> '+titles[1]+' leads')
    ax3.set_ylabel('lag distribution')
    ax3.set_title('time lag analysis')
    ax4 = ax3.twinx()
    ax4.plot(lagbins, ccf_stat, color=c)
    ax4.set_ylabel(titles_ccf_stat)

    if invert_colours is True:
        plt.style.use('dark_background')

    plt.tight_layout()
    if figure_title == 'show':
        plt.show()
    else:
        plt.savefig(figure_title,facecolor=fig.get_facecolor(),transparent=True)
    return ()
