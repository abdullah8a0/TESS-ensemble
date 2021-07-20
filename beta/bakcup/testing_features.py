
from operator import le
from astropy.stats.histogram import freedman_bin_width
import numpy as np
import astropy.stats as astat
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from numpy.core.numeric import roll
from numpy.ma.core import maximum_fill_value
import lcobj

lc = lcobj.lc_obj(21,1,1,1937,1913)




lc.plot()
'''
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
####### EXTRACT FREQUENCIES

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
