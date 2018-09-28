import numpy as np
import matplotlib.pylab as plt


#generate sensor data for bathroom light switch bedroom door, kitchen kettle

#True night time behaviour  1) bedroom door switch, 2) bathroom light, 3) bedroom door switch
#events 1, 2 and 3 will be separated by an average amount of time e.g 3minutes 

#add anomalies, sometimes kettle will go after bedroom door

#periodicity This will happen roughly 2x per night with some random distribution



#1 minute cadence
dt = 1.0

#assume 30days coverage
thi = 1.0*24*60



#night time hours
tstart_night = 0.0
tend_night   = 6.0*60
t = np.arange(0,thi,dt)
tday = 24.*60.






#set the model
nsense = 3
ibed = 0
ibath = 1
ikitchen = 2
nt = np.shape(t)[0]
y = np.zeros((nt,nsense))

tlow = t[0]
thi  = tlow + tend_night

tnow = t[0]
nbath = 2
sd = 0.5
tmean = 4.0
inight = 0
while tnow < t[-1]:
 print 'tnow',tnow
 for ib in range(nbath):
  tbed_out  = np.int(tnow + np.random.choice(np.int(tend_night), 1))
  tbath_on  = np.int(tbed_out + tmean + sd*np.random.randn(1))
  tbath_off = np.int(tbath_on + tmean + sd*np.random.randn(1))
  tbed_in   = np.int(tbath_off + tmean + sd*np.random.randn(1))
  print 'tbed_out,tbath_on,t_bath_off,tbed_in',tbed_out,tbath_on,tbath_off,tbed_in
  
  y[tbed_out:tbed_in,ibed] = 1
  y[tbath_on:tbath_off,ibath] = 1
  tnow      = tbed_in
 print tnow,'tnow'
 print ''
 tnow = tlow + inight*tday 
 inight = inight + 1 
 







#plot the results for each 'sensor'
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax1.plot(t,y[:,ibed],label='bed sensor')
ax2.plot(t,y[:,ibath],label='bathroom sensor')
ax3.plot(t,y[:,ikitchen],label='kitchen sensor')
plt.legend()	
plt.show()



