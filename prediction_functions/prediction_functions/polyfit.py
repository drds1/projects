import numpy as np
import matplotlib.pylab as plt
import prediction_functions.signal_decomp as vsd
from sklearn.metrics import r2_score
import scipy.optimize as so


# iteratively fit polynomials up to order, using BIC,AIC and reduced chi_squared to determine
# the optimum number of parameters.
def fit_search(x, y, sig = None, maxorder=8, xgrid=[], verbose=False):
    yg_med = []
    yg_lo = []
    yg_hi = []
    cov = []
    cisq = []
    cisq_red = []
    bic = []
    aic = []

    for i in range(1, maxorder):
        a = fit(x, y, sig, order=i, xgrid=xgrid)
        yg_med.append(a[0])
        yg_lo.append(a[1])
        yg_hi.append(a[2])
        cov.append(a[3])
        cisq.append(a[4])
        cisq_red.append(a[5])
        bic.append(a[6])
        aic.append(a[7])

    # sort by increasing fit metrics
    id_aic = np.argsort(aic)
    id_bic = np.argsort(bic)
    id_cisqred = np.argsort(np.abs(np.array(cisq_red) - 1.0))  # np.argsort(cisq_red)

    if verbose:
        print('best cisqred fit order:', id_cisqred[0])
        print('best aic fit order:', id_aic[0])
        print('best bic fit order:', id_bic[0])

    return (id_cisqred[0] + 1, id_aic[0] + 1, id_bic[0] + 1)




#define general polynomial for curve fitting function
def fit_func(x,*coeffs):
    y = np.polyval(coeffs,x)
    return y

# fit polynomial of a given order conf=0.05 for 95 % , 0.3173 for 1 sigma
def fit(xin, yin, sig=None, order=3, xgrid=[], confidence=0.3173, nits=20000, figure_title='', verbose=False):

    itp = np.argsort(xin)
    x = xin[itp]
    y = yin[itp]
    if sig is not None:
        if 0 in sig:
            sig = np.ones(len(x))


    # evaluate model on arbitrary time grid
    if (xgrid != []):
        xg = np.array(xgrid)
    else:
        xg = np.array(x)
    nxg = np.shape(xg)[0]

    # cannot fit trend with fewer points than polynomial order
    nx = np.shape(x)[0]
    oi = min(nx - 4, order)
    coeff_in = np.ones(oi+1)
    if (oi < 0):
        print('fit is degenerate, just using straight line')
        try:
            yg_med = np.ones(nxg) * y[0]
        except:
            yg_med = np.zeros(nxg)
        yg_lo = yg_med * 1
        yg_hi = yg_med * 1
        cov = np.zeros((1, 1))
        cisq = 0
        cisq_red = 0
        bic = 0
        aic = 0
        rmd = 0
        return (yg_med, yg_lo, yg_hi, cov, cisq, cisq_red, bic, aic, rmd)


    # compute and subtract pivot point (get better fit with reduced correlated parameter uncertainties)


    if sig is None:
        w = np.ones(nx)
        xp = np.mean(x)
    else:
        w = 1./sig
        xp = np.sum(x / sig ** 2) / np.sum(1. / sig ** 2)
    if xp != xp:
        xp = np.mean(x)



    #coefs, cov = np.polyfit(x - xp, y, w = w, deg=oi, full=False, cov=True)
    coefs,cov = so.curve_fit(fit_func,(x-xp),y,sigma = 1./w,p0=coeff_in,absolute_sigma=True)
    #coefs,cov,r2  = vsd.constituents_polyfit(x-xp,y,order=oi)
    yg_itp = np.sum(np.array([coefs[-1 - ip] * (x - xp) ** ip for ip in range(oi + 1)]), axis=0)
    var = np.var(yg_itp - y)#np.sum((ymod_itp - y) ** 2) / nx
    sd  = np.sqrt(var)
    ndof = nx - oi - 1

    if sig is None:
        cov_expansion = np.sum((yg_itp - y) ** 2) /ndof
        cov = cov * cov_expansion
        if verbose is True:
            print('sigma is none')
            print('ts variance = ',var)
            print('covariance matrix',cov)
            print('number of points, number of parms, ndof',nx,oi,ndof)
            print('covariance expansion',cov_expansion)
            print('theoretical reduced chi squared',np.sum((yg_itp - y)**2)/cov_expansion/ndof   )

    r2 = r2_score(y,yg_itp)




    # monte carlo sample error snake
    c_mvn = np.random.multivariate_normal(coefs, cov, size=nits)
    yg = np.zeros((nxg, nits))
    xgxp_grid = np.array([(xg - xp) ** ip for ip in range(oi + 1)])

    for iteration in range(nits):
        cnow = c_mvn[iteration, :]
        yg_temp = np.dot(xgxp_grid.T, cnow[::-1])
        yg[:, iteration] = yg_temp



    # compute final model and uncertainties

    if (nits > 1):
        pc = [confidence / 2 * 100, 50., (1. - confidence / 2) * 100]
        yg_med_conf = np.percentile(yg, pc, axis=1).T
        yg_lo = yg_med_conf[:, 0]
        yg_med = yg_med_conf[:, 1]
        yg_hi = yg_med_conf[:, 2]
    else:
        yg_med = np.sum(np.array([coefs[-1 - ip] * (xg - xp) ** ip for ip in range(oi + 1)]), axis=0)
        yg_lo,yg_hi = yg_med - sd, yg_med + sd




    # compute model evaluation stats (big numbers for lots of parameters or poor fits)
    # reduced chisquared
    cisq = np.sum((yg_itp - y)**2 / var)
    ndof = nx - oi + 1
    cisq_red = cisq / ndof

    # autocoorelation
    rmd = np.corrcoef(yg_itp, y)[0, 1]

    # Bayesian information criterion
    bic = cisq + (oi + 1) * np.log(nx)

    # Akaike information criterion
    aic = cisq + 2 * (oi + 1)

    # results summary
    if verbose is True:
        print('pivot point', xp)
        print('fit coefficients', coefs)
        print('error envelope separation ',np.median(yg_hi - yg_lo))
        print('x min max', x.min(), x.max())
        print('y min max', y.min(), y.max())
        print('')
        print('fit results for poilynomial order: ', oi)
        print('')
        print('Smaller numbers better')
        print('reduced chisquare: ', cisq_red)
        print('aic: ', aic)
        print('bic: ', bic)
        print('chisquare: ', cisq)
        print('correlation coefficient: ', rmd)
        print()
        print()



    if (figure_title != ''):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(x, y)
        ax1.plot(xg, yg_med, label='model')
        ax1.plot(x, yg_itp, label='interpolated')
        #ax1.fill_between(xg, yg_lo, yg_hi, alpha=0.3, label='old uncertainties')
        #ax1.fill_between(xg, yg_lo_mod, yg_hi_mod, alpha=0.3, label='new uncertainties')
        plt.legend()
        if (figure_title == 'show'):
            plt.show()
        else:
            plt.savefig(figure_title)

    return (yg_med, yg_lo, yg_hi, cov, cisq, cisq_red, bic, aic, rmd, r2)

##test with fake data
#
#import numpy as np
#import matplotlib.pylab as plt
#import glob
#import pandas as pd
#import datetime
#import statsmodels.api as sm
#
#combine = 0
#frequency = 1./30. #fake 6 month signal
#f2 = 1./60
#tlo = 0.0
#thi = 365
#tref = 0.0#thi/2
#dt = 1.0
#lab = ''
#amp1 = 5.0
#amp2 = 10.0
#amps = [amp1,amp2]
#freqs = [frequency,f2]
#poly = [0,0,5.e-5,0.0]
#tfclo = thi
#tfchi = thi + 365
##generate synthetic time series for test
#t = np.arange(tlo,thi,dt)
#noise = 4.0
#x = 0.1*t + 0.5 + np.random.randn(np.shape(t)[0])*noise
#
#fit(t, x, sig=None, order=1, xgrid=[], confidence=0.3173, nits=20000, figure_title='show', verbose=False)
#
#
###add trend final -10.0 indicates amplitude at the end of the time sequence
## grad = 1.0
## y1   = 10.0
## noiseamp = 0.5
##
## pc_forecast = 0.4 #forecast ahead 40% the original length of the time series
##
###specify the confidence interval with the alpha argument
###e.g alpha = 0.05 is the 95pc confidence interval (2 sigma)
###alpha = 0.32 is the 1 sigma confidence interval (68pc)
## alpha_conf = 0.32#0.05
##
##
#
# def trend(t,poly,tref,freqs,amps):
# nt = np.shape(t)[0]
# npoly = len(poly)
# namps = len(amps)
# xtot = []
# for i in range(nt):
#  trend = np.sum([poly[ip]*(t[i]-tref)**ip for ip in range(npoly)] )
#  seasonal = np.sum([amps[ip]*np.sin(2*np.pi*freqs[ip]*t[i]) for ip in range(namps)] )
#  xtot.append(trend + seasonal)
#
# return(np.array(xtot))
# 
# 
#

# signal = trend(t,poly,tref,freqs,amps)
##grad*(x-t[-1]) + y1
# nt = np.shape(t)[0]
# noise = np.random.randn(nt)*noiseamp
# signal = signal + noise
#
# t = np.arange(100)
# x = np.random.randn(100)*1.0 + 5.0 #+ 2.3*t + 3*t**2
#
#
# tgrid = np.arange(-20,120,0.1)
# fit_search(t,x,maxorder=8,xgrid=tgrid)

##signal = trend(t,poly,tref,freqs,amps) 
###grad*(x-t[-1]) + y1
##nt = np.shape(t)[0]
##noise = np.random.randn(nt)*noiseamp
##signal = signal + noise
##
##t = np.arange(100)
##x = np.random.randn(100)*1.0 + 5.0 #+ 2.3*t + 3*t**2
##
##
##tgrid = np.arange(-20,120,0.1)
##fit_search(t,x,maxorder=8,xgrid=tgrid)
