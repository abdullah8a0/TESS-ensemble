import numpy as np
import statsmodels.tsa.stattools as stattools
import scipy as sp
from scipy import stats
from sklearn.metrics import r2_score
import concurrent.futures
import astropy.stats as astat
import plot_lc
from plot_lc import get_coords_from_path
import os
from lcobj import LCMissingDataError, LCOutOfBoundsError, gen_path, LC

def extract_vector_feat_from_tag(tag):
    sector,cam,ccd,col,row = tag
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
    for cam,ccd in np.ndindex((4,4)):
        cam +=1
        ccd +=1
        print(sector,cam,ccd)

        path = gen_path(sector,cam,ccd,0,0)[:-6]

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
            for feat in results:
                if feat is not None and np.all(np.isfinite(feat)) and  not np.any(np.isnan(feat)):
                    Data.append(feat)
            Data = np.array(Data)
            with open(f'Features\\features{sector}_{cam}_{ccd}_v.txt', 'w') as file:
                np.savetxt(file,Data,fmt = '%1.5e',delimiter=',' )





                            
if __name__ == '__main__':
    extract_vector_features(35)
    extract_vector_features(36)
    extract_vector_features(37)
    extract_vector_features(38)
    extract_vector_features(33)
    extract_vector_features(32)
    extract_vector_features(34)