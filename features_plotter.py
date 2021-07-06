import numpy as np
from lcobj import base,get_sector_data
import matplotlib.pyplot as plt
from scipy import stats

sector = 32

tag, data = next(get_sector_data(sector, 's'))

TOI = [
    (1, 3, 1953, 1208),
(2, 1, 1787, 866) ,
(2, 2, 1425, 411) ,
(2, 2, 1416, 1162),
(2, 3, 1109, 1645),
(2, 3, 1192, 362),
(2, 3, 216, 2030),
(3, 1, 1778, 738),
(3, 1, 137, 399),
(3, 1, 1032, 739),
(3, 1, 1584, 1125),
(3, 2, 100, 1095),
(3, 2, 1178, 2001),
(3, 2, 1103, 606),
(3, 4, 195, 1152),
(4, 1, 1752, 1579),
(4, 2, 70, 1634),
(4, 3, 957, 1933),
(4, 3, 1505, 1708),
(4, 3, 509, 256),
(4, 3, 1877, 1600),
(4, 3, 1044, 1626),
(4, 3, 939, 1874)
]



def TOI_color(data):
    ret = []
    for i in range(len(data)):
        point = tag[i]
        if tuple(point.astype('int32').tolist()) in TOI:
            ret.append('red')
        else:
            ret.append('green')
    return np.array(ret)

'''
feat = np.array([amp,better_amp,med,mean,std,slope,r,skew,max_slope,\
beyond1std, delta_quartiles, flux_mid_20, *flux_mid_35*, flux_mid_50, \
flux_mid_65, flux_mid_80, *cons*, slope_trend, var_ind, med_abs_dev, \
H1, R21, R31, Rcs, l , med_buffer_ran, perr, StetK, p_ander, day_of_i])
'''

feature = data[:,-1]

for i in range(len(data)):
    point = tag[i][2:4]
    if tuple(point.astype('int32')) in TOI:
        print(point, feature[i])


plt.ion()
fig,ax = plt.subplots()

import lcobj

from scipy.stats import linregress
ax.scatter(range(len(feature)),feature, s= 1,picker=5, c=TOI_color(feature))
def onpick(event):
    ind = event.ind
    ccd_point = tag[ind][0]
    coords = (int(ccd_point[0]),int(ccd_point[1]),int(ccd_point[2]), int(ccd_point[3]))
    fig1,ax1 = plt.subplots()
    lc = lcobj.LC(sector,*coords)
    print(coords)
    slope, c = linregress(lc.time,lc.flux)[:2]
    #print(data_raw[ind][0][-1])

    #lc.pad_flux()
    #lc.make_FFT()
    ax1.scatter(lc.time,lc.flux,s=0.5)
    #ax1.scatter(lc.phase_space,lc.flux,s=0.5)
    #ax1.scatter(lc.time, lc.smooth_flux,s=0.5)
    ax1.scatter(lc.time, slope*lc.time+c,s=0.5)


fig.canvas.mpl_connect('pick_event', onpick)
plt.show(block = False)
input()





#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()

#plt.scatter(range(len(feature)),scaler.fit_transform(feature.reshape(-1,1)), s= 1)
#plt.show()
