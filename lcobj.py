import numpy as np
from astropy import stats as astat
import matplotlib.pyplot as plt
import pylab
from scipy import stats,signal
import math
from sklearn.ensemble import IsolationForest


def set_base(loc):
    global base
    base = loc


#working_folder = "C:\\Users\\saba saleemi\\Desktop\\UROP\\TESS\\"

def gen_path(sector,cam,ccd,col,row):
    # Generates path to specific file
    file_name = 'lc_'+str(col)+'.'+str(row)

    ccd_path_raw = ["s","_transient_lc\\sector","\\cam","_ccd","\\lc_transient_pipeline\\"]
    sector2 = str(sector) if sector > 9 else '0'+str(sector)
    ccd_path = ccd_path_raw[0]+sector2+ccd_path_raw[1]+sector2+ccd_path_raw[2]+str(cam)+ccd_path_raw[3]+str(ccd)+ccd_path_raw[4]

    try:
        file_path = base + ccd_path + file_name
    except NameError:
        from run_classif import base
        set_base(base)
        file_path = base + ccd_path + file_name
    return file_path

class LCOutOfBoundsError(Exception):
    pass
class LCMissingDataError(Exception):
    pass

class LC(object):
    def __init__(self, sector, cam, ccd, col, row):
        mask = {
            35: (2268.50, 2272.045),
        }
        

        try:
            assert 44 <= int(col) <= 2097
            assert 0 <= int(row) <= 2047 
        except AssertionError:
            raise LCOutOfBoundsError

        self.path = gen_path(sector,cam,ccd,col,row)
        self.sector = int(sector)
        self.cam = int(cam)
        self.ccd = int(ccd)
        self.coords = (int(col),int(row))
        lc_data = np.genfromtxt(self.path)

        try:
            assert len(lc_data.shape) == 2
        except AssertionError:
            raise LCMissingDataError("The flux file is almost empty")


        self.flux_unclipped = -lc_data[:, 1]
        self.time_unclipped = lc_data[:, 0]
        self.error_unclipped = lc_data[:, 2]
        self.bg_unclipped = -lc_data[:, 6]
        
        bg_clip = astat.sigma_clip(self.bg_unclipped,sigma=3)

        self.flux = np.array([self.flux_unclipped[i] for i in np.ma.nonzero(bg_clip)[0]]) 
        self.time = np.array([self.time_unclipped[i] for i in np.ma.nonzero(bg_clip)[0]])
        self.error = np.array([self.error_unclipped[i] for i in np.ma.nonzero(bg_clip)[0]])
        self.bg = np.array([self.bg_unclipped[i] for i in np.ma.nonzero(bg_clip)[0]])

        if len(self.flux) < 10:
            raise LCMissingDataError("The flux file has less than 10 entries")
        if sector in mask.keys():
            masking_arr = np.logical_or(self.time < mask[sector][0],self.time > mask[sector][1])

            self.flux   = self.flux[masking_arr]
            self.time   = self.time[masking_arr]
            self.error  = self.error[masking_arr]
            self.bg     = self.bg[masking_arr]   


        self.N = len(self.flux)
        self.mean = np.mean(self.flux)
        self.std = np.std(self.flux)

        try:
            assert self.N > 60
        except AssertionError:
            raise LCMissingDataError("The flux file has less than 60 entries")

        # Smoothing using SavGol Filter
        try:

            self.smooth_flux = signal.savgol_filter(self.flux,min((1|int(0.05*self.N),61)), 3)
        except:
            raise LCMissingDataError
        self.linreg = stats.linregress(self.time,self.flux)[0:3]  #(slope,c,r)

        self.is_padded = False
        self.is_FFT = False         # Flags whether the instance has these attributes
        self.is_percentiles = False  

        self.normed_flux = (self.flux - self.mean)/np.max(np.abs(self.flux-self.mean))
        self.normed_time = self.time/np.max(self.time)
        self.normed_smooth_flux = (self.smooth_flux - self.mean)/np.max(np.abs(self.smooth_flux-self.mean))

    def plot(self, flux = None, time = None, show_bg = True, show_err = False,show_smooth=False,scatter=[]):
        flux = self.flux if flux is None else flux
        time = self.time if time is None else time

        #fig = pylab.gcf()
        #fig.canvas.manager.set_window_title('Figure_'+str(self.coords[0])+'_'+str(self.coords[1]))
        plt.xlabel("Time (Days)")
        plt.ylabel("Raw Flux")
        plt.scatter(time,flux,s=0.5)
        if show_bg:
            plt.scatter(time,self.bg,s=0.1)
        if show_err:
            plt.errorbar(time, flux, yerr=self.error, fmt ='o' )
        if show_smooth:
            plt.scatter(time,self.smooth_flux,s=0.1)
        for data in scatter:
            try:
                time_s,flux_s = data
                plt.scatter(time_s,flux_s,s=0.1)
            except:
                plt.scatter(time,data,s=0.1)
        plt.show()
        return self

    def remove_outliers(self):
        outlier_detector = IsolationForest(n_estimators=200,max_samples=1000,random_state=np.random.RandomState(314),contamination=0.01)  # dynamic percentage?
        forest = outlier_detector.fit_predict(list(zip(b:=self.normed_flux,a:=np.arange(0,1,1/len(self.normed_flux)))))
        #forest = outlier_detector.fit_predict(self.normed_flux.reshape(-1,1))
        inliers = np.ma.nonzero(forest==1)[0]
        self.normed_flux = self.normed_flux[inliers]
        self.normed_time = self.normed_time[inliers]
        self.flux        = self.flux[inliers]
        self.time        = self.time[inliers]
        self.error       = self.error[inliers]
        self.bg          = self.bg[inliers]
        self.N           = inliers.size
        self.mean        = np.mean(self.flux)
        self.std         = np.std(self.flux)
        self.median      = np.median(self.flux)
        try: 
            self.smooth_flux = signal.savgol_filter(self.flux,min((1|int(0.05*self.N),61)), 3)
        except:
            raise LCMissingDataError

        if self.N < 60:
            raise LCMissingDataError

        return self
    def make_percentiles(self):
        self.percentiles = {i:np.percentile(self.flux,i) for i in range(0,101,1)}
        self.is_percentiles = True

    def flatten(self, smooth = None):
        flux = self.smooth_flux if smooth is None else self.flux

        fit = signal.savgol_filter(self.flux, 201 , 1)
        self.flat =  flux - fit
        return self

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

        freq_f, fft_f = freq[50:upper], np.abs(fft[50:upper])       # Final results        
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
        plt.scatter(np.arange(len(self.smooth_flux)), self.smooth_flux, s=0.5)
        plt.show()

class TagFinder:
    def __init__(self,tags_original):

        tags = np.copy(tags_original)

        tags = tags.tolist()
        tags = sorted(enumerate(tags), key =lambda x:x[1])
        map = {i:tags[i][0] for i in range(len(tags))}
        tags = np.array([x[1] for x in tags])                 # reorder feat with tags
        self.tags = tags
        self.map = map
    
    def find(self,tag):
        return self.map[self.bin_search(np.array(tag))]



    def gt(self,x,y):
        return self.lt(y,x)

    def bin_search(self,tag) -> int:
        tags = self.tags
        upper = tags.shape[0]-1
        lower = 0
        while lower <= upper:
            mid = (upper + lower)//2
            if self.lt(tags[mid], tag):
                lower = mid + 1
            elif self.gt(tags[mid] , tag):
                upper = mid - 1
            else:
                return mid
        raise Exception(f"Tag is not in the data : {tuple(tag)}")

    def lt(self,x,y):
        if (x==y).all():
            return False
        idx = np.where((x!=y))[0][0]
        #print(x[idx])
        return x[idx]< y[idx]

def get_sector_data(sectors,t,verbose=True):
    #include sectors as 3rd output
    assert t in ('s','v')

    if not hasattr(sectors, '__iter__'):
        sectors = [sectors]       


    for sector in sectors:
        sector2 = str(sector) if int(sector) > 9 else '0'+str(sector)
        flag = True
        for cam,ccd in np.ndindex((4,4)):
            cam +=1
            ccd +=1
            if verbose:
                print("Loading:", sector,cam,ccd)
            if flag:
                data_raw =np.genfromtxt(f"Features\\features{sector2}_{cam}_{ccd}_{t}.txt", delimiter=',')
                if data_raw.all():
                    continue
                flag = False 
            else:
                data_raw_ccd = np.genfromtxt(f"Features\\features{sector2}_{cam}_{ccd}_{t}.txt", delimiter=',')
                if data_raw_ccd.all():
                    continue
                if len(data_raw_ccd.shape) == 1:
                    continue
                if data_raw_ccd.size!=0:
                    data_raw = np.concatenate((data_raw, data_raw_ccd))

        tags, data = data_raw[::,1:5].astype(int), data_raw[::,5:]  
        if t=='s' and 0:
            data = np.delete(data,[],1)

        yield tags,data


def get_coords_from_path(file_path):
    # returns coords from any file path
    i = file_path.rfind('lc_')
    d = file_path.rfind('.')

    x = file_path[i+3:d]
    y = file_path[d+1:]
    return (x,y)
if __name__ == '__main__':
    
    set_base("C:\\Users\\saba saleemi\\Desktop\\UROP\\TESS\\transient_lcs\\unzipped_ccd\\")                 # Change this to the folder where your sxx_transient_lcs is located