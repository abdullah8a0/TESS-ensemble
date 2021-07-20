import matplotlib.pyplot as plt
import numpy as np
from lcobj import base as base, gen_path
from plot_lc import get_coords_from_path
import os

data_raw = np.genfromtxt(base + "py_code\\count_transients_s1-34.txt")

sector = 7

tran_x, tran_y = data_raw[:,-2], data_raw[:,-1]

tran_x, tran_y = tran_x[np.ma.nonzero(data_raw[:,0] == sector)], tran_y[np.ma.nonzero(data_raw[:,0] == sector)]

plt.scatter(tran_x,tran_y, marker='s', label='Known transients')

TOI21 = [
    (190,709),
    (303,176),
    (472,862),
    (526,1002),
    (772,994),
    (965,1603),
    (1196,1855),
    (2070,1674),
    (1675,235),
    (1095,540)
]

TOI6 = [
    (611,2017),
    (610,897),
    (902,905),
    (1182,1232)
]



x,y= [], []
for cam,ccd in np.ndindex((4,4)):
    cam +=1
    ccd +=1

    sector2 = str(sector) if sector > 9 else '0'+str(sector)
    path = gen_path(sector,cam,ccd,0,0)[:-6]
    with os.scandir(path) as entries:
        for i,entry in enumerate(entries):
            try:
                coords = tuple([int(i) for i in get_coords_from_path(entry.name)])
            except ValueError:
                continue
            x.append(coords[0])
            y.append(coords[1])
            if coords in TOI6:
                print(cam,ccd,coords)
plt.scatter(x,y,s=1)
plt.legend()
plt.show()