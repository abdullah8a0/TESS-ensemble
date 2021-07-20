from numpy.core.shape_base import block
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
import lcobj
import numpy as np

import matplotlib.pyplot as plt
base = lcobj.base

TOI = [
    (1182,1232),
    (1363,1908),
    (886,1540),
    (945,1892),
    (1055,1543),
    (1491,708)
]


sector, cam, ccd = 6,4,1

sector2 = str(sector) if sector > 9 else '0'+str(sector)


data_raw = np.genfromtxt(base + f"py_code\\Features\\features{sector2}_{cam}_{ccd}.txt", delimiter=',')

data = data_raw[::,2:]

scaler = StandardScaler()
data_norm = scaler.fit_transform(data)

#TNSE START

from sklearn.manifold import TSNE

def f(data):
    ret = []
    for i in range(len(data)):
        point = data_raw[i][0:2]
        if (int(point[0]), int(point[1])) in TOI:
            ret.append(1)
        else:
            ret.append(0)
    return np.array(ret)


plt.ion()
fig,ax = plt.subplots()

data_tsne = TSNE(n_components=2).fit_transform(data_norm)
ax.scatter(data_tsne[:,0], data_tsne[:, 1], s = 5, picker=5, c= f(data_tsne))
def onpick(event):
    ind = event.ind
    ccd_point = data_raw[ind][0][0:2]
    coords = (int(ccd_point[0]), int(ccd_point[1]))
    fig1,ax1 = plt.subplots()
    lc = lcobj.lc_obj(sector,cam,ccd,*coords)
    print(coords)
    ax1.scatter(lc.time,lc.flux,s=0.5)
    #ax.show(block = False)



fig.canvas.mpl_connect('pick_event', onpick)
plt.show(block = False)
input()




#new = SelectKBest(k=10).fit_transform(data_norm,np.array([None]*len(data)))
#print(new)

#print(data_norm)

'''
pca = PCA(n_components=2)



print("starting")
pca.fit(data_norm)
print("done!")

print(pca.explained_variance_ratio_)
print(pca.singular_values_)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

transformed_data = pca.transform(data_norm)
#print(transformed_data[:,0])
print(np.argmax(transformed_data[:,0]))
ax.scatter(transformed_data[:,0], transformed_data[:,1],s = 5)
ax.grid()
plt.show()
'''

