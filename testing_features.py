
from operator import le
from astropy.stats.histogram import freedman_bin_width
import numpy as np
import astropy.stats as astat
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from numpy.core.defchararray import find
from numpy.core.numeric import roll
from numpy.ma.core import maximum_fill_value
from scipy.signal import find_peaks
import sklearn
import lcobj
from lmfit.models import StepModel, ConstantModel

lc = lcobj.lc_obj(6,2,4,1182,1232)

# 6 1 1 1001 1198 for shadowing
# 6 1 1 1280 1564 for fanning

lc.plot()
lc.smooth_plot()
#plt.scatter(lc.time,[0]*lc.N)
#plt.scatter(lc.time,lc.flux, s= 1)
#plt.show()






















#exit()

from sklearn.metrics import r2_score            # STEP FUNCTION FIT #################################

step_init = lc.smooth_flux[0]
step_fin = lc.smooth_flux[-1]
center = np.argmax(np.abs(lc.smooth_flux[1:] - lc.smooth_flux[:-1]))

k = len(lc.smooth_flux)
center = center if k> center else k
center = round(center) if center > 0 else 0

fit = np.concatenate((np.array([step_init]*center), np.array([step_fin]*(k-center))))

perr = r2_score(lc.smooth_flux,fit)
print(np.log(1/(1-perr))) 
#alpha = np.arange(len(out.best_fit))-center
#fit = Amp*np.min([1,np.max([0,alpha])])
plt.scatter(np.arange(k),fit, s=0.5)
plt.scatter(np.arange(k),lc.smooth_flux, s=0.5)
plt.show()

exit()

import statsmodels.tsa.stattools as stattools           # AUTO COREELATION ###############################

lags = 100

Auto_Cor = stattools.acf(lc.flux, nlags=lags)                
l = next((i for i,val in enumerate(Auto_Cor) if val < np.exp(-1)), None)

while l is None:

    lags += 100

    Auto_Cor = stattools.acf(lc.flux, nlags=lags)                
    l = next((i for i,val in enumerate(Auto_Cor) if val < np.exp(-1)), None)

Auto_Cor_len = l


'''                                                                 ######## LOMB SCARGLE ###################
rolling_avg = np.convolve(lc.flux, np.ones(100), 'valid')
plt.scatter(np.arange(len(rolling_avg)), rolling_avg, s=0.5)
plt.show()
exit()
dflux_dt = np.abs(rolling_avg[1:]-rolling_avg[:-1])/(rolling_avg[1:]-rolling_avg[:-1])
slope_rolling_average = np.convolve(dflux_dt, np.ones(50), 'valid')      # Rolling average of 5
plt.scatter(np.arange(len(slope_rolling_average)),slope_rolling_average, s= 0.5)
plt.show()

print('normed LS')
freq, pow = LombScargle(lc.time,lc.normed_flux).autopower()
plt.plot(freq,pow)
plt.show()

max_freq = freq[np.argmax(pow)]
print('peak:', np.max(pow))
T = 1/max_freq
print("The period to observation ratio is:", T/(1/48))

temp = lc.time - lc.time[0]
new_time = []
for i in range(len(temp)):
    while temp[i] > T:
        temp[i] -= T

plt.scatter(np.array(temp), lc.normed_flux, s= 0.5)
plt.show()

exit()
#print('peak', peak)
#exit()

'''
####### EXTRACT FREQUENCIES                                         ########## FFT ########################

print('padded fft')
lc.pad_flux()
lc.make_FFT()
lc.plot_FFT()
lc.plot_phase()
print(lc.significant_fequencies)
freq, pow =lc.fft_freq

best_fit_fft = lc.significant_fequencies
z_score = (best_fit_fft[:,1]-np.mean(pow))/np.std(pow)

print(z_score)





exit()
len_fft = len(lc.padded_flux)
fft = np.fft.fft(lc.padded_flux)[:len_fft//2]#*1/len_fft

perday = 48 if lc.sector <= 26 else 144
delta_t = 1/perday
freq = np.fft.fftfreq(len_fft, delta_t)[:len_fft//2]
upper = np.abs(freq - 200).argmin()
plt.plot(freq[:upper],np.abs(fft)[:upper])
plt.show()







exit()

'''
max_freq = freq[np.argmax(fft)]
print("peak:", max_freq)
T = 1/max_freq

print("The period to observation ratio is:", T/(1/48))
temp = lc.time - lc.time[0]
new_time = []
for i in range(len(temp)):
    while temp[i] > T:
        temp[i] -= T

#temp = np.linspace(0,T,T/48)

ind = np.argpartition(fft, -10)[-10:]
top_10_freq = ind
print(top_10_freq)
plt.scatter(np.arange(len(lc.padded_flux))%186,lc.padded_flux, s= 0.5)
plt.show()
'''
