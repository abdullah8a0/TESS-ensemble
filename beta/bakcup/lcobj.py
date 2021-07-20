import numpy as np
from astropy import stats as astat
import matplotlib.pyplot as plt
from numpy.core.shape_base import block
import pylab
import scipy as sp
from scipy import stats 
import math
from astropy.timeseries import LombScargle

base = "C:\\Users\\saba saleemi\\Desktop\\UROP\\TESS\\"                 # Change this to the folder where your transient_lcs is located
ccd_path_raw = ["transient_lcs\\unzipped_ccd\\s","_transient_lc\\sector","\\cam","_ccd","\\lc_transient_pipeline\\"]


def gen_path(sector,cam,ccd,col,row):
    # Generates path to specific file
    file_name = 'lc_'+str(col)+'.'+str(row)

    sector2 = str(sector) if sector > 9 else '0'+str(sector)
    ccd_path = ccd_path_raw[0]+sector2+ccd_path_raw[1]+sector2+ccd_path_raw[2]+str(cam)+ccd_path_raw[3]+str(ccd)+ccd_path_raw[4]

    file_path = base + ccd_path + file_name
    return file_path

class LCOutOfBoundsError(Exception):
    pass

class lc_obj(object):
    def __init__(self, sector, cam, ccd, col, row):
        try:
            assert 44 <= int(col) <= 2097
            assert 0 <= int(row) <= 2047 
        except AssertionError:
            raise LCOutOfBoundsError

        self.path = gen_path(sector,cam,ccd,col,row)
        self.sector = sector
        self.cam = cam
        self.ccd = ccd
        self.coords = (int(col),int(row))
        lc_data = np.genfromtxt(self.path)

        assert len(lc_data) > 10

        self.flux_unclipped = lc_data[:, 1]
        self.time_unclipped = lc_data[:, 0]
        self.error_unclipped = lc_data[:, 2]
        self.bg_unclipped = lc_data[:, 6]
        
        bg_clip = astat.sigma_clip(self.bg_unclipped,sigma=3)

        self.flux = np.array([self.flux_unclipped[i] for i in np.ma.nonzero(bg_clip)[0]]) 
        self.time = np.array([self.time_unclipped[i] for i in np.ma.nonzero(bg_clip)[0]])
        self.error = np.array([self.error_unclipped[i] for i in np.ma.nonzero(bg_clip)[0]])
        self.bg = np.array([self.bg_unclipped[i] for i in np.ma.nonzero(bg_clip)[0]])

        if len(self.flux) < 10:
            raise TypeError

        self.N = len(self.flux)
        self.mean = np.mean(self.flux)
        self.std = np.std(self.flux)

        self.is_padded = False
        self.is_FFT = False         # Flags whether the instance has these attributes
        self.is_percentiles = False  

        self.normed_flux = (self.flux - self.mean)/np.max(np.abs(self.flux))

    def plot(self, flux = None, time = None, show_bg = True, show_err = False):
        flux = self.flux if flux is None else flux
        time = self.time if time is None else time

        #fig = pylab.gcf()
        #fig.canvas.manager.set_window_title('Figure_'+str(self.coords[0])+'_'+str(self.coords[1]))
        print("here")
        plt.xlabel("Time (Days)")
        plt.ylabel("True Flux (/10^6)")
        plt.scatter(time,flux,s=0.5)
        if show_bg:
            plt.scatter(time,self.bg,s=0.1)
        if show_err:
            plt.errorbar(time, flux, yerr=self.error, fmt ='o' )
        plt.show()

    def make_percentiles(self):
        self.percentiles = {i:np.percentile(self.flux,i) for i in range(0,101,1)}
        self.is_percentiles = True

    def pad_flux(self):
        perday = 48 if self.sector <= 26 else 144
        bins = int((self.time[-1] - self.time[0])*perday)
        
        stat = np.nan_to_num(stats.binned_statistic(self.time, self.normed_flux, bins=bins)[0])

        pow2 = math.ceil(math.log2(len(stat)) + math.log2(6))
        pow2 = 1
        while 2**pow2 < len(stat)*6:
            pow2+=1
        padded_flux = np.zeros(2**pow2)
        padded_flux[0:len(stat)] = stat

        self.padded_flux = padded_flux
        self.is_padded = True
    
    def make_FFT(self):
        assert self.is_padded
        len_fft = len(self.padded_flux)

        fft = np.fft.fft(self.padded_flux)[:len_fft//2]#*1/len_fft

        self.is_FFT = True
        perday = 48 if self.sector <= 26 else 144
        delta_t = 1/perday
        
        freq = np.fft.fftfreq(len_fft, delta_t)[:len_fft//2]
        upper = np.abs(freq - 200).argmin()

        freq_f, fft_f = freq[:upper], np.abs(fft[:upper])       # Final results        
        self.fft_freq = (freq_f, fft_f)
        # Top 5 (freq,pow)

        ind = np.argpartition(fft_f, -5)[-5:]
        self.significant_fequencies = np.array(sorted(zip(freq_f[ind], fft_f[ind]), reverse=True, key = lambda elem: elem[1])) 

        T = 1/self.significant_fequencies[0][0]

        phase = self.time - self.time[0]
        for i in range(len(phase)):
            while phase[i] > T:
                phase[i] -= T

        self.phase_space = phase        # Folded time space

    def plot_FFT(self):
        assert self.is_FFT
        freq, pow = self.fft_freq
        plt.scatter(freq,pow,s=5)
        plt.show()        

    '''
    def make_periodogram(self):
        self.LS_object = LombScargle(self.time, self.normed_flux, self.error)
        freq, pow = self.LS_object.autopower()    # Check with supervisor to set appropriiate minimum and max frequencies
        self.periodogram = (freq,pow)
        self.computed_freq = abs(freq[np.argmax(pow)])
        self.computed_freq_power = np.max(pow)
        self.is_LombScargle = True
    
        T = 1/self.computed_freq

        phase = self.time - self.time[0]
        for i in range(len(phase)):
            while phase[i] > T:
                phase[i] -= T

        self.phase_space = phase        # Folded time space
    '''


    def plot_phase(self):
        assert self.is_FFT

        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Figure_'+str(self.coords[0])+'_'+str(self.coords[1])+'_Phase(FFT)')
        plt.xlabel("Phase")
        plt.ylabel("True Flux (/10^6)")

        plt.scatter(self.phase_space,self.flux,s=0.5)
        plt.show()

    def smooth_plot(self):
        n = 100
        rolling_avg = np.convolve(self.flux, 1/n*np.ones(n), 'valid')
        plt.scatter(np.arange(len(rolling_avg)), rolling_avg, s=0.5)
        plt.show()

