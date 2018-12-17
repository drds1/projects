import numpy as np
import mylcgen as mlc
import matplotlib.pylab as plt

#produce white noise test
test = np.random.randn(1000)

#generate light curve from custom data base
tlo = 0
thi = 1000
dat = mlc.mylcgen(datfile='',p0=1.0,f0=0.1,a=-2,b=-2,tlo=tlo,thi=thi,dt=1.0,ploton=0,iseed=-1,meannorm = -1., sdnorm = -1.0)


#inject anomaly
test[250:500] = test[250:500] + 5.
dat[250:500,1] = dat[250:500,1] + 700


#diagnostic plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(dat[:,0],dat[:,1])
plt.show()



#luminol test
import luminol
import luminol.anomaly_detector
import luminol.correlator

detector = luminol.anomaly_detector.AnomalyDetector(dat[:,1])
anomalies = detector.get_anomalies()

if anomalies:
    time_period = anomalies[0].get_time_window()
    correlator = luminol.correlator.Correlator(dat[:,1], ts2, time_period)
    
print(correlator.get_correlation_result().coefficient)










 

