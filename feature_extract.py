import numpy as np
import statsmodels.tsa.stattools as stattools
import concurrent.futures
from scipy import stats
from sklearn.metrics import r2_score
import os
from lcobj import LC, LCMissingDataError, LCOutOfBoundsError, gen_path,get_coords_from_path
from pathlib import Path

import numpy as np
import statsmodels.tsa.stattools as stattools
import scipy as sp
from scipy import stats
from sklearn.metrics import r2_score
import concurrent.futures
import astropy.stats as astat
from lcobj import get_coords_from_path
import os
from lcobj import LCMissingDataError, LCOutOfBoundsError, gen_path, LC
from pathlib import Path

def extract_scalar_feat_from_tag(tag):

    try:
        lc = LC(*tag)
        #lc_orig = LC(*tag)
        lc.remove_outliers() #<= try to get this to work
    except TypeError:
        print("empty: ", tag[-2:])
        return None
    except LCOutOfBoundsError:
        print("out of bounds: ", tag[-2:])
        return None
    except LCMissingDataError:
        print("TOO LITTLE DATA:", tag[-2:])
        return None

    # Peak-to-Peak /2
    amp = np.ptp(lc.flux)/2
    
    # Median
    med = np.median(lc.flux)

    # Mean 
    mean = lc.mean

    # std
    std = lc.std

    basic_var = std/mean        # simple variability

    # Small Kurtosis

    S = sum(((lc.flux - mean) / std) ** 4)
    n = lc.N
    #c1 = (n*(n+1))/((n-1)*(n-2)*(n-3))
    #c2 = 3*(n-1)**2/((n-2)*(n-3))

    #small_k = c1*S-c2

    ###############
    wei = [1/err**2 if err!=0 else 0 for err in lc.error]
    wmean = np.average(lc.flux,weights=wei)

    wvar = sum((lc.flux - wmean)**2)
    wstd = np.sqrt(wvar/(lc.N-1))

    num_1std = np.sum(np.logical_or(lc.flux > wmean + wstd, lc.flux < wmean - wstd))

    beyond1std = num_1std/lc.N              
    
    # data points beyand 1std

    # Robust Kurtosis

    sigmap = (np.sqrt(lc.N/(lc.N-1)))* (lc.flux - wmean)/lc.error

    StetK = (1 / np.sqrt(lc.N * 1.0) * np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap ** 2)))


    # linear fit
    slope, c, r = lc.linreg

    #skew
    skew = stats.skew(np.squeeze(lc.flux))

    # max slope

    dflux_dt = (lc.flux[1:]-lc.flux[:-1])/(lc.time[1:]-lc.time[:-1])
    max_slope = np.max(np.abs(dflux_dt))

    slope_trend = np.sum(np.sign(dflux_dt))/lc.N                    # fraction of increasing - decreasing slopes

    slope_rolling_average = np.convolve(dflux_dt, np.ones(5), 'valid')      # Rolling average of 5
    std_slope_rolling_average = np.std(slope_rolling_average)
    max_slope_rolling_average = np.max(slope_rolling_average)

    # INSTRUMENT ARTIFACTS DETECTION




    #### RTS via step fitting
    step_init = lc.smooth_flux[0]
    step_fin = lc.smooth_flux[-1]
    center = np.argmax(np.abs(lc.smooth_flux[1:] - lc.smooth_flux[:-1]))

    half_1 = np.arange(0,center)
    half_2 = np.arange(center,len(lc.smooth_flux))

    #m1,_,r1 = stats.linregress(lc.normed_time[half_1],lc.normed_smooth_flux[half_1])[:3]
    #m2,_,r2 = stats.linregress(lc.normed_time[half_2],lc.normed_smooth_flux[half_2])[:3]

    k = len(lc.smooth_flux)
    center = center if k> center else k
    center = center if center > 0 else 0

    step_center = (step_fin + step_init)/2
    band_width = np.min(np.abs(lc.flux - step_center))



    fit = np.concatenate((np.array([step_init]*center), np.array([step_fin]*(k-center))))
    
    perr = r2_score(lc.smooth_flux,fit)

    #perr = abs(m1-m2) + abs(r1-r2)
    
    # Neumann's Variability Index (for unevenly spaced data)

    #w = 1 / (lc.time[1:]-lc.time[:-1])**2
    #w_mean = np.mean(w)

    #var_lc = np.var(lc.flux)

    #S1 = sum(w * (lc.flux[1:]-lc.flux[:-1])**2)
    #S2 = sum(w)

    #var_ind = (w_mean * np.power(lc.time[lc.N - 1] - lc.time[0], 2) * S1 / (var_lc * S2 * lc.N ** 2))

    var_ind = 1/((lc.N - 1)*std**2) * np.sum((lc.flux[1:] - lc.flux[:-1])**2)

    #check of clusteral anomalies

    # Median Deviation

    med_abs_dev = np.median(np.abs(lc.flux - med))

    # Median Buffer

    med_buffer_ran = np.sum(np.logical_and(lc.flux < med + amp/5, lc.flux > med - amp/5 ))/lc.N

    # Kim's Consecutives

    num_cons = 0
    for i in range(lc.N-2):
        if not any((mean - 2*std <val< mean + 2*std) for val in lc.flux[[i,i+1,i+2]]):
            num_cons += 1
    cons = num_cons/(lc.N - 2)


    # Anderson Darling Test

    ander = stats.anderson(lc.flux)[0]
    p_ander = 1/(1.0 + np.exp(-10*(ander-0.3)))




    # FLUX PERCENTILE FEATURES
    lc.make_percentiles()

    ### percentile ratios:

    Flux_5_95 = lc.percentiles[95] - lc.percentiles[5]

    Flux_40_60 = lc.percentiles[60] - lc.percentiles[40]

    Flux_32_68 = lc.percentiles[68] - lc.percentiles[32]

    Flux_25_75 = lc.percentiles[75] - lc.percentiles[25]

    Flux_17_83 = lc.percentiles[83] - lc.percentiles[17]
    
    Flux_10_90 = lc.percentiles[90] - lc.percentiles[10]

    flux_mid_20 = Flux_40_60/Flux_5_95
    flux_mid_35 = Flux_32_68/Flux_5_95
    flux_mid_50 = Flux_25_75/Flux_5_95
    flux_mid_65 = Flux_17_83/Flux_5_95
    flux_mid_80 = Flux_10_90/Flux_5_95

    ##


    per5 = lc.percentiles[5]
    per95 = lc.percentiles[95]

    upper = np.median(lc.flux[lc.flux > per95])
    lower = np.median(lc.flux[lc.flux < per5])

    better_amp = (upper - lower)/2      # Better Amplitude

    delta_quartiles = lc.percentiles[75] - lc.percentiles[25]


    # Auto-Correlation

    lags = 100

    Auto_Cor = stattools.acf(lc.flux, nlags=lags, fft=False)                
    l = next((i for i,val in enumerate(Auto_Cor) if val < np.exp(-1)), None)

    while l is None:
        lags += 100

        Auto_Cor = stattools.acf(lc.flux, nlags=lags,fft=False)                
        l = next((i for i,val in enumerate(Auto_Cor) if val < np.exp(-1)), None)


    # FFT Features
    lc.pad_flux() 
    lc.make_FFT()

    freq, pow  = lc.fft_freq

    best_fit_fft = lc.significant_fequencies
    z_score = (best_fit_fft[:,1]-np.mean(pow))/np.std(pow)

    H1 = best_fit_fft[0][1] # Maybe amplitude is meaningless?
    R21 = best_fit_fft[1][1]/ best_fit_fft[0][1]
    R31 = best_fit_fft[2][1]/ best_fit_fft[0][1]

    # Rcs:

    S = np.cumsum(lc.flux - mean)*1/(lc.N * std)
    Rcs =  np.max(S) - np.min(S)
    
    #days of interest
    
    granularity = 1.0           # In days
    bins = granularity*np.arange(27)
    bin_map = np.digitize(lc.time-lc.time[0], bins)

    interesting_d = []
    total_d = 0
    for bin in bins:
        dp_in_bin = np.ma.nonzero(bin_map == bin+1)
        flux, time = lc.smooth_flux[dp_in_bin], lc.time[dp_in_bin]
        _, ind = np.unique(time, return_index=True)
        flux, time = flux[ind], time[ind]

        if flux.size == 0:
            interesting_d.append(0)
            continue

        if np.mean(flux) > std + mean:
            interesting_d.append(1)
        else:
            interesting_d.append(0)
        total_d +=1
    days_of_i = sum(interesting_d)/total_d 



    data_start = lc.flux[:30]
    data_end = lc.flux[-30:]

    slope_trend_start = (len(np.where(np.diff(data_start)>0)[0]) - len(np.where(np.diff(data_start)<=0)[0]))/30
    slope_trend_end = (len(np.where(np.diff(data_end)>0)[0]) - len(np.where(np.diff(data_end)<=0)[0]))/30
    
    
    ###############
    feat = np.array([*tag,better_amp,med,mean,std,slope,r,skew,max_slope,\
    beyond1std, delta_quartiles, flux_mid_20,flux_mid_35, flux_mid_50, \
    flux_mid_65, flux_mid_80, cons, slope_trend, var_ind, med_abs_dev, \
    H1, R21, R31, Rcs, l , med_buffer_ran, np.log(1/(1-perr)),band_width, StetK, p_ander, days_of_i,slope_trend_start,slope_trend_end])

    return feat.astype('float32')


def extract_scalar_features(sector):
    all_data = []
    for cam,ccd in np.ndindex((4,4)):
        cam +=1
        ccd +=1
        print(sector,cam,ccd)

        path = Path(str(gen_path(sector,cam,ccd,0,0))[:-6])
        if not Path.exists(path):
            Path.mkdir(path,parents=True, exist_ok=True)

        tags = []
        with os.scandir(path) as entries:
            for entry in entries:
                if not entry.name.startswith('.') and entry.is_file():
                    if entry.name[:3] != 'lc_':
                        continue
                    tag = (sector,cam,ccd,*get_coords_from_path(entry.name))

                    tags.append(tag)

        # Parallel Processing Start
        with concurrent.futures.ProcessPoolExecutor() as executer:
            results = executer.map(extract_scalar_feat_from_tag,tags)
            Data = []
            for i,feat in enumerate(results):
                if i%100 == 0:
                    print(i)
                if feat is not None and np.all(np.isfinite(feat)) and not np.any(np.isnan(feat)):
                    Data.append(feat)
            Data = np.array(Data)
            all_data = Data if all_data == [] else np.concatenate((all_data,Data))
    with open(Path(f'Features/{sector}_scalar.csv'), 'w') as file:
        np.savetxt(file,all_data,fmt = '%1.5e',delimiter=',')


def extract_vector_feat_from_tag(tag):
    try:
        lc = LC(*tag)
        lc.remove_outliers()
    except TypeError:
        print("empty: ", tag[-2:])
        return None
    except LCOutOfBoundsError:
        print("out of bounds: ", tag[-2:])
        return None
    except LCMissingDataError:
        print("TOO LITTLE DATA:", tag[-2:])
        return None

    granularity = 1.0           # In days
    bins = granularity*np.arange(27)
    bin_map = np.digitize(lc.normed_time-lc.normed_time[0], bins)

    feat = []
    for bin in bins:# range(1,np.max(bin_map)+1):
        dp_in_bin = np.ma.nonzero(bin_map == bin+1)
        flux, time = lc.normed_flux[dp_in_bin], lc.normed_time[dp_in_bin]
        _, ind = np.unique(time, return_index=True)
        flux, time = flux[ind], time[ind]

        if flux.size in {0,1}:
            slope , r = 0,1
        else:
            slope, _, r = stats.linregress(time,flux)[:3]  #(slope,c,r)

    ###############
        feat += [slope, r**2]
    
    feat = np.array([*tag,*feat])
    return feat.astype('float32')

def extract_vector_features(sector):
    all_data = []
    for cam,ccd in np.ndindex((4,4)):
        cam +=1
        ccd +=1
        print(sector,cam,ccd)

        path = Path(str(gen_path(sector,cam,ccd,0,0))[:-6])

        tags = []
        with os.scandir(path) as entries:
            for i,entry in enumerate(entries):
                if not entry.name.startswith('.') and entry.is_file():
                    if entry.name[:3] != 'lc_':
                        continue
                    tag = (sector,cam,ccd,*get_coords_from_path(entry.name))

                    tags.append(tag)

        with concurrent.futures.ProcessPoolExecutor() as executer:
            results = executer.map(extract_vector_feat_from_tag,tags)
            Data = []
            for i,feat in enumerate(results):
                if i%100==0:
                    print(i)
                if feat is not None and np.all(np.isfinite(feat)) and  not np.any(np.isnan(feat)):
                    Data.append(feat)
            Data = np.array(Data)

            all_data = Data if all_data == [] else np.concatenate((all_data,Data))
    with open(Path(f'Features/{sector}_vector.csv'), 'w') as file:
        np.savetxt(file,all_data,fmt = '%1.5e',delimiter=',')


def extract_signat_feat_from_tag(tag):

    try:
        lc = LC(*tag)
        lc.remove_outliers()
    except TypeError:
        print("empty: ", tag[-2:])
        return None
    except LCOutOfBoundsError:
        print("out of bounds: ", tag[-2:])
        return None
    except LCMissingDataError:
        print("TOO LITTLE DATA:", tag[-2:])
        return None
    granularity = 1.0/3           # In days
    bins = granularity*np.arange(round(27/granularity))
    bin_map = np.digitize(lc.time-lc.time[0], bins)

    signature = []
    total_d = 0
    for bin in bins:#range(1,np.max(bin_map)+1):
        dp_in_bin = np.ma.nonzero(bin_map == round(bin/granularity)+1)
        flux, time = lc.flux[dp_in_bin], lc.time[dp_in_bin]
        _, ind = np.unique(time, return_index=True)
        flux, time = flux[ind], time[ind]

        if flux.size == 0:
            signature.append(0)
            continue
        if np.mean(flux) > lc.std + lc.mean:
            signature.append(1)
        else:
            signature.append(0)

        total_d +=1
    feat = np.array([*tag,*signature])
    return feat.astype('float32')

def extract_signat_features(sector):
    all_data = []
    for cam,ccd in np.ndindex((4,4)):
        cam +=1
        ccd +=1
        print(sector,cam,ccd)

        path = Path(str(gen_path(sector,cam,ccd,0,0))[:-6])

        tags = []
        with os.scandir(path) as entries:
            for i,entry in enumerate(entries):
                if not entry.name.startswith('.') and entry.is_file():
                    if entry.name[:3] != 'lc_':
                        continue
                    tag = (sector,cam,ccd,*get_coords_from_path(entry.name))

                    tags.append(tag)

        with concurrent.futures.ProcessPoolExecutor() as executer:
            results = executer.map(extract_signat_feat_from_tag,tags)
            Data = []
            for i,feat in enumerate(results):
                if i%100==0:
                    print(i)
                if feat is not None and np.all(np.isfinite(feat)) and  not np.any(np.isnan(feat)):
                    Data.append(feat)
            Data = np.array(Data)

            all_data = Data if all_data == [] else np.concatenate((all_data,Data))
    with open(Path(f'Features/{sector}_signat.csv'), 'w') as file:
        np.savetxt(file,all_data,fmt = '%1.5e',delimiter=',')


 
if __name__ == '__main__':
    extract_scalar_features(38)
    #extract_scaler_features(32)
    #extract_scaler_features(39)
    #extract_scaler_features(40)
    #extract_scaler_features(41)
    #extract_scaler_features(42)