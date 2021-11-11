import os
import numpy as np
from lcobj import gen_path
from run_classif import base
from shutil import copyfile
from pathlib import Path

working_folder = Path('/Users/abdullah/Desktop/UROP/Tess/sector_data/transient_lcs')
def spec(x):
    return int(round(float(x)))

sector = 42
transients = np.genfromtxt(working_folder / "known_transient_lc/count_transients_s1-42.txt",dtype= str)
sec_t = transients[np.ma.nonzero(transients[:,0]==str(sector))][:,(3,7,9,10,11,12)]#3,6,8,9,10,11
tags, data = sec_t[:,1], np.vectorize(spec)(sec_t[:,(0,2,3,4,5)])


transi = np.array([[1,1,1,1]],dtype='int64')
for cam,ccd in np.ndindex((4,4)):
    cam +=1
    ccd +=1
    print(sector,cam,ccd)

    sector2 = str(sector) if sector > 9 else '0'+str(sector)
    path2 = Path(str(gen_path(sector,cam,ccd,0,0))[:-6])
    #path2 = Path('/Users/abdullah/Desktop/UROP/Tess/local_code/py_code/transient_data/transient_lc')
    path1 = working_folder / f"known_transient_lc/sector{sector2}/cam{cam}_ccd{ccd}/lc_discovery/"
    name = []
    try:
        with os.scandir(path1) as entries:
            #print('scanning')
            for entry in entries:
                if entry.name[-3:] in {'png'}:
                    continue
                entry_tag = entry.name[3:-8]
                try:
                    entry_data = data[np.where(tags==entry_tag)][0]
                except IndexError:
                    print(f"{entry_tag} not found in {cam} {ccd}")
                    continue
                if entry_data[0] > 18.5:
                    continue
                print(tuple(entry_data[1:]))
                transi = np.concatenate((transi, np.array(entry_data[1:]).reshape(1,entry_data.shape[0]-1).astype('int64')),axis = 0)
                os.remove(path2/f"lc_{entry_data[-2]}.{entry_data[-1]}")
    except FileNotFoundError:
        continue
print(transi)
#with open(working_folder / f"{sector}_transients.txt", 'w') as f:
#    np.savetxt(f,transi[1:,:])