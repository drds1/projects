from prediction_functions.apply_lag import *
from prediction_functions.polyfit import *
def correlation_figure(x,y,lag,
                       order = 1,xylabs=('x','y'),figure_title='show',
                       bin = True,
                       axfig = None,
                       global_title='correlation analysis'):
    """
    Compute the cooreltion between x and y (with a given lag) and plot the result.

    :param x: Input x values (numpy 1D array)
    :param y: Input y values (numpy 1D array)
    :param lag: Integer number of time steps -ve indicates y leads x
    :param order: polynomial order to fit to data (default 1)
    :param xylabs: labels for x and y axis (tuple(xlab,ylab))
    :param figure_title: the output figure title (if 'show' then plot to screen)
    :param bin: If true, bin the x,y data into 10 x bins before fitting model.
        The displayed points will be unaltered but the binned points will be used to calculate
        the uncertainty. This is all because linear regression uncertainty envelopes
        screw up when you have lots of points sampled around the same x value as here.
        Default error envelopes too small. Rebinning goes at least part of the way to fixing this.
    :return Either plot figure to screen (if figure_title = 'show') or save to file:
    """
    #identify overlapping elements and shift by lag
    if lag == 0:
        x_common,y_common = x,y
        correlation_coefficient = np.corrcoef(x_common, y_common)[0, 1]
    else:
        x_common, y_common, correlation_coefficient = apply_lag(x,y,lag)

    #make figure
    xmin,xmax,xsd = np.min(x_common),np.max(x_common),np.std(x_common)
    xgrid = np.linspace(xmin-xsd,xmax+xsd,len(x_common)*10)

    #produced binned results for reliable error envelopes
    if bin is True:
        from scipy.stats import binned_statistic as bs
        y_common_fit, bin_edges, bin_number  = bs(x_common,values=y_common,statistic='mean')
        y_common_sig, bin_edges, bin_number = bs(x_common, values=y_common, statistic=np.std)
        x_common_fit = (bin_edges[1:]+bin_edges[:-1])/2
    else:
        y_common_fit = y_common
        x_common_fit = x_common
        y_common_sig = None



    if axfig is None:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
    else:
        ax1,fig = axfig

    ax1.scatter(x_common, y_common, label=None, color='b')

    try:
        yg_med, yg_lo, yg_hi, cov, cisq, cisq_red, bic, aic, rmd, r2 =\
            fit(x_common_fit, y_common_fit, sig=y_common_sig, order=order,
            xgrid=xgrid, confidence=0.3173, nits=20000, figure_title='', verbose=False)

        ax1.plot(xgrid,yg_med,label=None,color='cyan')
        ax1.fill_between(xgrid,yg_lo,yg_hi,alpha=0.4,color='cyan')
    except:
        print('correlation figure.py... polyfit failed')

    ax1.set_xlabel(xylabs[0])
    ax1.set_ylabel(xylabs[1])
    ax1.annotate(r'$r_c = '+np.str(np.round(correlation_coefficient,2))+'$',
                  (0.95, 0.95), xycoords='axes fraction',
                 horizontalalignment='right', color='cyan')
    ax1.set_title(global_title)
    plt.style.use('dark_background')

    if figure_title == 'show':
        plt.show()
    elif figure_title == '':
        pass
    else:
        plt.savefig(figure_title)
    if axfig is None:
        return()
    else:
        return(ax1,fig)

