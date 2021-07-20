import os
import numpy as np
from lcobj import working_folder,gen_path
from shutil import copyfile

def spec(x):
    return int(round(float(x)))

transients = np.genfromtxt(working_folder +"py_code\\count_transients_s1-34.txt",dtype= str)
sec_32_t = transients[np.ma.nonzero(transients[:,0]=='33')][:,(3,6,8,9,10,11)]
tag, data = sec_32_t[:,1], np.vectorize(spec)(sec_32_t[:,(0,2,3,4,5)])


sector = 33

for cam,ccd in np.ndindex((4,4)):
    cam +=1
    ccd +=1
    #print(sector,cam,ccd)

    sector2 = str(sector) if sector > 9 else '0'+str(sector)
    path2 = gen_path(sector,cam,ccd,0,0)[:-6]
    path1 = working_folder + f"py_code\\s33_knowns\\cam{cam}_ccd{ccd}"
    name = []
    with open(working_folder + "py_code\\33_transients.txt", 'w') as f, os.scandir(path1) as entries:
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