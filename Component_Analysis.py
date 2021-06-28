from TOI_gen import TOI_list
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import lcobj
import numpy as np
from sklearn.feature_selection import VarianceThreshold

import matplotlib.pyplot as plt
base = lcobj.base

TOI6 = [
    (1182,1232),
    (1363,1908),
    (886,1540),
    (945,1892),
    (1055,1543),
    (1491,708)
]

TOI21 = [
    (467,870)
]


sector = 6

TOI = TOI6

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

data = data_raw[::,4:]


scaler = StandardScaler()

for i in data_raw:
    if not np.all(np.isfinite(i)):
        print(i)
print(np.all(np.isfinite(data)))

#data_best = VarianceThreshold(threshold=0.005).fit_transform(data/np.mean(data))

data_norm = scaler.fit_transform(data)

#### PCA START
print("---Reducing Dimensionality using PCA---")

pca = PCA(n_components=12)

pca.fit(data_norm)

print("Explained Variance:",sum(pca.explained_variance_ratio_), pca.explained_variance_ratio_)

transformed_data = pca.transform(data_norm)

#Clustering start
import hdbscan

print("---Dimesionality Reduced. Starting Cluster using HDBSCAN---")

clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=12,metric='euclidean')       # BEST is 15,12 cluster size

clusterer.fit(transformed_data)
print("Number of cluster:", np.max(clusterer.labels_)+1)
print([np.count_nonzero(clusterer.labels_ == i) for i in range(-1,1+np.max(clusterer.labels_))])

anomalies = np.ma.nonzero(clusterer.labels_ == -1)

#from sklearn.ensemble import IsolationForest
#forest = IsolationForest(random_state=314,contamination=0.4).fit_predict(transformed_data)
#TNSE START

print("---Clustering done. Visualising using t-SNE---")

#for i in range(len(data)):
#    point = data_raw[i][2:4]
#    if tuple(point.astype('int32')) in TOI:
#        print(point, clusterer.labels_[i], forest[i])


from sklearn.manifold import TSNE

def color_cluster(data,clusterer):
    return clusterer.labels_

def color_TOI(data):
    ret = []
    for i in range(len(data)):
        point = data_raw[i][2:4]
        if tuple(point.astype('int32')) in TOI:
            ret.append('red')
            print(point.astype('int32').tolist())
        else:
            ret.append('green')
    return np.array(ret)


#def color_anomaly(data):
#    return forest


plt.ion()
fig,ax = plt.subplots()

data_tsne = TSNE(n_components=2).fit_transform(transformed_data)        ############# transformed or normed
ax.scatter(data_tsne[:,0], data_tsne[:, 1], s = 5, picker=5, c= color_cluster(data_tsne,clusterer))       # what to color? 
def onpick(event):
    ind = event.ind
    ccd_point = data_raw[ind][0][0:4]
    coords = (int(ccd_point[0]),int(ccd_point[1]),int(ccd_point[2]), int(ccd_point[3]))
    fig1,ax1 = plt.subplots()
    lc = lcobj.lc_obj(sector,*coords)
    print(coords)
    #print(data_raw[ind][0][-1])
    print(clusterer.labels_[ind])
    ax1.scatter(lc.time,lc.flux,s=0.5)
    #ax.show(block = False)



fig.canvas.mpl_connect('pick_event', onpick)
#plt.show(block = False)                            ############# CHANGE THIS BACK
input()












anom_data =  transformed_data[anomalies]
anom_cluster = hdbscan.HDBSCAN(min_cluster_size=14, min_samples=12,metric='euclidean')
anom_cluster.fit(anom_data)


plt.ion()
fig2,ax2 = plt.subplots()

data_tsne = TSNE(n_components=2).fit_transform(anom_data)        ############# transformed or normed
ax2.scatter(data_tsne[:,0], data_tsne[:, 1], s = 5, picker=5, c= color_cluster(data_tsne,anom_cluster))       # what to color? 
def onpick(event):
    ind = event.ind
    ccd_point = data_raw[ind][0][0:4]
    coords = (int(ccd_point[0]),int(ccd_point[1]),int(ccd_point[2]), int(ccd_point[3]))
    fig1,ax1 = plt.subplots()
    lc = lcobj.lc_obj(sector,*coords)
    print(coords)
    ax1.scatter(lc.time,lc.flux,s=0.5)



fig2.canvas.mpl_connect('pick_event', onpick)
plt.show(block = False)
input()









