import os
import numpy as np
from lcobj import gen_path
from shutil import copyfile

working_folder = 'C:\\Users\\saba saleemi\\Desktop\\UROP\\TESS\\'

def spec(x):
    return int(round(float(x)))

sector = 38
transients = np.genfromtxt(working_folder +"py_code\\count_transients_s1-40.txt",dtype= str)
sec_32_t = transients[np.ma.nonzero(transients[:,0]==str(sector))][:,(3,7,9,10,11,12)]#3,6,8,9,10,11
tag, data = sec_32_t[:,1], np.vectorize(spec)(sec_32_t[:,(0,2,3,4,5)])



for cam,ccd in np.ndindex((4,4)):
    cam +=1
    ccd +=1
    #print(sector,cam,ccd)

    sector2 = str(sector) if sector > 9 else '0'+str(sector)
    path2 = gen_path(sector,cam,ccd,0,0)[:-6]
    path1 = working_folder + f"py_code\\beta\\kt_33-39\\kt_33-39\\known_transients_s{sector}\\cam{cam}_ccd{ccd}"
    name = []
    with open(working_folder + f"py_code\\{sector}_transients.txt", 'w') as f, os.scandir(path1) as entries:
        for entry in entries:
            if entry.name[-8:] in {'_cleaned','aned.png'}:
                continue
            entry_tag = entry.name[3:]
            try:
                entry_data = data[np.where(tag==entry_tag)][0]
            except IndexError:
                continue
            if entry_data[0] > 18.5:
                continue
            print(tuple(entry_data[1:]))
            np.savetxt(f,np.array(entry_data[1:]).reshape(1,entry_data.shape[0]-1)) 
            #input()
            copyfile(path1 + f"\\{entry.name}",path2+f"lc_{entry_data[-2]}.{entry_data[-1]}")