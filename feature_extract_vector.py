import numpy as np
import statsmodels.tsa.stattools as stattools
import scipy as sp
from scipy import stats
from sklearn.metrics import r2_score
import astropy.stats as astat
import plot_lc
from plot_lc import get_coords_from_path
import os
from lcobj import LCMissingDataError, LCOutOfBoundsError, gen_path, LC


sector = 6


for cam,ccd in np.ndindex((4,4)):
    cam +=1
    ccd +=1
    print(sector,cam,ccd)

    sector2 = str(sector) if sector > 9 else '0'+str(sector)
    path = gen_path(sector,cam,ccd,0,0)[:-6]
    with open(f'Features\\features{sector2}_{cam}_{ccd}_v.txt', 'w') as file, os.scandir(path) as entries:
        for i,entry in enumerate(entries):
            if i%10 == 0:
                print(i)
            if not entry.name.startswith('.') and entry.is_file():
                if entry.name[:3] != 'lc_':
                    continue
                file_path = path + entry.name
                try:
                    lc = LC(sector,cam,ccd,*get_coords_from_path(entry.name))
                except TypeError:
                    print("empty: ", entry.name)
                    continue
                except LCOutOfBoundsError:
                    print("out of bounds: ", entry.name)
                    continue
                except LCMissingDataError:
                    print("TOO LITTLE DATA:", entry.name)
                    continue

                granularity = 1.0           # In days
                bins = granularity*np.arange(int((lc.time[-1]-lc.time[0])/granularity) + 2)
                bin_map = np.digitize(lc.time-lc.time[0], bins)

                for bin in range(1,np.max(bin_map)+1):
                    dp_in_bin = np.ma.nonzero(bin_map == bin)
                    flux, time = lc.flux[dp_in_bin], lc.time[dp_in_bin]
                    _, ind = np.unique(time, return_index=True)
                    flux, time = flux[ind], time[ind]

                    if flux.size in {0,1}:
                        continue

                    slope, _, r = stats.linregress(time,flux)[:3]  #(slope,c,r)

                ###############
                    feat = np.array([slope, r])
                    coords = plot_lc.get_coords_from_path(file_path)
                    if np.all(np.isfinite(feat)) and  not np.any(np.isnan(feat)):
                        file.write(f'{cam},{ccd},{coords[0]},{coords[1]},')
                        feat = feat.reshape(1,feat.shape[0])
                        np.savetxt(file,feat,fmt = '%1.5e',delimiter=',' )
                    else:
                        print("Bad Features on coords:", coords)
                        print("Features:", feat)
                        print("data:",flux,time)
                        exit()