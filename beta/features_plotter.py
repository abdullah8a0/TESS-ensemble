from TOI_gen import TOI_list
import numpy as np
from lcobj import get_sector_data, set_base
import matplotlib.pyplot as plt
from scipy import stats


base = "C:\\Users\\saba saleemi\\Desktop\\UROP\\TESS\\transient_lcs\\unzipped_ccd\\" # Forced value of base
set_base(base)

sector = 32
TOI = TOI_list(sector)

import cluster_anomaly 
TOI = cluster_anomaly.sector_32_must_detects
tag, data = next(get_sector_data(sector, 's',verbose=False))


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
flux_mid_65, flux_mid_80, *cons*, slope_trend, var_ind(near 2 means normal dist), med_abs_dev, \
H1, R21, R31, Rcs, l , med_buffer_ran, perr, StetK, p_ander, day_of_i])
'''

feature = data[:,0]/data[:,1]

for i in range(len(data)):
    point = tag[i]
    if tuple(point.astype('int32')) in TOI:
        print(point, feature[i])


plt.ion()
fig,ax = plt.subplots()

import lcobj

from scipy.stats import linregress

try:
    num_feat = feature.shape[1]
    for i in range(num_feat):
        ax.scatter(range(feature.shape[0]),feature[:,i], s= 1,picker=5)
except IndexError:
    num_feat = 1
    ax.scatter(range(feature.shape[0]),feature[:], s= 1,picker=5, c= TOI_color(feature))

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
    ax1.scatter(lc.time, lc.smooth_flux,s=0.5)
    #ax1.scatter(lc.time, slope*lc.time+c,s=0.5)


fig.canvas.mpl_connect('pick_event', onpick)
plt.show(block = False)
input()





#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()

#plt.scatter(range(len(feature)),scaler.fit_transform(feature.reshape(-1,1)), s= 1)
#plt.show()
