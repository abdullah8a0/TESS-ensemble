import numpy as np
from lcobj import base
import matplotlib.pyplot as plt
sector = 6


sector2 = str(sector) if sector > 9 else '0'+str(sector)

flag = True
for cam,ccd in np.ndindex((4,4)):
    cam +=1
    ccd +=1
    print("Loading:", sector,cam,ccd)
    if flag:
        data_raw =np.genfromtxt(base + f"py_code\\Features\\features{sector2}_{cam}_{ccd}.txt", delimiter=',')
        if data_raw.all():
            continue
        flag = False 
    else:
        data_raw_ccd = np.genfromtxt(base + f"py_code\\Features\\features{sector2}_{cam}_{ccd}.txt", delimiter=',')
        if data_raw_ccd.all():
            continue
        data_raw = np.concatenate((data_raw, data_raw_ccd))

TOI = [
    (1182,1232),
    (1363,1908),
    (886,1540),
    (945,1892),
    (1055,1543),
    (1491,708)
]



def TOI_color(data):
    ret = []
    for i in range(len(data)):
        point = data_raw[i][2:4]
        if tuple(point.astype('int32').tolist()) in TOI:
            ret.append('red')
        else:
            ret.append('green')
    return np.array(ret)
'''
feat = np.array([amp,better_amp,med,mean,std,slope,r,skew,max_slope,\
beyond1std, delta_quartiles, flux_mid_20, *flux_mid_35*, flux_mid_50, \
flux_mid_65, flux_mid_80, *cons*, slope_trend, var_ind, med_abs_dev, \
H1, R21, R31, Rcs, l , med_buffer_ran, perr, StetK, p_ander])
'''
feature = data_raw[:,-3]

for i in range(len(data_raw)):
    point = data_raw[i][2:4]
    if tuple(point.astype('int32')) in TOI:
        print(point, feature[i])

plt.scatter(range(len(feature)),feature, s= 1, c=TOI_color(feature))
plt.show()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#plt.scatter(range(len(feature)),scaler.fit_transform(feature.reshape(-1,1)), s= 1)
plt.show()
