import numpy as np
import statsmodels.tsa.stattools as stattools
import scipy as sp
from scipy import stats
from sklearn.metrics import r2_score
import astropy.stats as astat
import plot_lc
from plot_lc import get_coords_from_path
import os
from lcobj import LCMissingDataError, LCOutOfBoundsError, gen_path, lc_obj


sector = 6

for cam,ccd in np.ndindex((4,4)):
    cam +=1
    ccd +=1
    print(sector,cam,ccd)

    sector2 = str(sector) if sector > 9 else '0'+str(sector)
    path = gen_path(sector,cam,ccd,0,0)[:-6]
    with open(f'Features\\features{sector2}_{cam}_{ccd}.txt', 'w') as file, os.scandir(path) as entries:
        for i,entry in enumerate(entries):
            if i%10 == 0:
                print(i)
            if not entry.name.startswith('.') and entry.is_file():

                if entry.name[:3] != 'lc_':
                    continue
                file_path = path + entry.name
                try:
                    lc = lc_obj(sector,cam,ccd,*get_coords_from_path(entry.name))
                except TypeError:
                    print("empty: ", entry.name)
                    continue
                except LCOutOfBoundsError:
                    print("out of bounds: ", entry.name)
                    continue
                except LCMissingDataError:
                    print("TOO LITTLE DATA:", entry.name)
                    continue

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
                c1 = (n*(n+1))/((n-1)*(n-2)*(n-3))
                c2 = 3*(n-1)**2/((n-2)*(n-3))

                small_k = c1*S-c2

                ###############

                wmean = np.average(lc.flux,weights=1/lc.error**2)

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

                slope_trend = np.sum(dflux_dt/np.abs(dflux_dt))/lc.N                    # fraction of increasing - decreasing slopes

                slope_rolling_average = np.convolve(dflux_dt, np.ones(5), 'valid')      # Rolling average of 5
                std_slope_rolling_average = np.std(slope_rolling_average)
                max_slope_rolling_average = np.max(slope_rolling_average)

                # INSTRUMENT ARTIFACTS DETECTION




                #### RTS via step fitting
                step_init = lc.smooth_flux[0]
                step_fin = lc.smooth_flux[-1]
                center = np.argmax(np.abs(lc.smooth_flux[1:] - lc.smooth_flux[:-1]))

                k = len(lc.smooth_flux)
                center = center if k> center else k
                center = center if center > 0 else 0

                fit = np.concatenate((np.array([step_init]*center), np.array([step_fin]*(k-center))))
                
                perr = r2_score(lc.smooth_flux,fit)
                
                # Neumann's Variability Index

                var_ind = 1/((lc.N - 1)*std**2) * np.sum((lc.flux[1:] - lc.flux[:-1])**2)

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
                
                
                
                
                ###############
                feat = np.array([amp,better_amp,med,mean,std,slope,r,skew,max_slope,\
                beyond1std, delta_quartiles, flux_mid_20,flux_mid_35, flux_mid_50, \
                flux_mid_65, flux_mid_80, cons, slope_trend, var_ind, med_abs_dev, \
                H1, R21, R31, Rcs, l , med_buffer_ran, np.log(1/(1-perr)), StetK, p_ander])
                coords = plot_lc.get_coords_from_path(file_path)
                if np.all(np.isfinite(feat)) and  not np.any(np.isnan(feat)):
                    file.write(f'{cam},{ccd},{coords[0]},{coords[1]},')
                    feat = feat.reshape(1,feat.shape[0])
                    np.savetxt(file,feat,fmt = '%1.5e',delimiter=',' )
                else:
                    print("Bad Features on coords:", coords)
                    print("Features:", feat)