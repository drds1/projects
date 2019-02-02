from sklearn import decomposition
import numpy as np
from statsmodels.tsa.api import VAR, DynamicVAR



#initialising sequence for existing AR code
#ar_mad = RLI()
#ar_mad.lagrange = lagrange
#ar_mad.response_timeseries = x_new[:idsplit]
#ar_mad.dates_global = dates[:idsplit]
#for i in range(ncomponents):
#    ar_mad.add_ar_component(xc_new[i][:idsplit],
#                            name=self.x_components_name[i], dates=dates[:idsplit])


class VAR():



if __name__ == '__main__':
    font = {'family': 'normal',
            #'weight': 'bold',
            'size': 6}
    import matplotlib
    matplotlib.rc('font',**font)

    lag = 30.0
    wide = 1.0

    n = 400
    time = np.arange(n)
    signal = 10
    noise  = 0.2
    season = 0.000025 * time ** 2
    npredict = 100
    y    = signal*np.sin(2*np.pi*time/90) + noise*np.random.randn(n)


    # construct mock time series and include impulse response lag function
    tau      = np.arange(np.int(-n/2)+1,np.int(n/2)-1,1)
    response = np.exp(-0.5 * (tau - lag) ** 2 / wide ** 2) / (2 * np.pi * wide ** 2)
    echo     = scipy.convolve(y,response,mode='same')