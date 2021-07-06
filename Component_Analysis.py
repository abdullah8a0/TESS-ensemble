#from TOI_gen import TOI_list
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
    (4, 1, 467,870),
    (3, 4, 190,709)
]

TOI32 =[
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

sectors = [21]
TOI = TOI21


data_gen = lcobj.get_sector_data(sectors,'s',verbose=False)
tags, data = next(data_gen)
# Try removing p_adner
scaler = StandardScaler()

data_norm = scaler.fit_transform(data)

#### PCA START
print("---Reducing Dimensionality using PCA---")

pca = PCA(n_components=15)   #12,13 best, 14 prev, 15 WORKS
pca.fit(data_norm)

#print("Explained Variance:",sum(pca.explained_variance_ratio_), pca.explained_variance_ratio_)

#print(pca.singular_values_)
#print(pca.components_)

transformed_data = pca.transform(data_norm)

#Clustering start

import hdbscan

print("---Dimesionality Reduced. Starting Cluster using HDBSCAN---")

clusterer = hdbscan.HDBSCAN(min_cluster_size=8,metric='euclidean',prediction_data=True)       # BEST is 15,12 cluster size, 19 previous, 7 prev, 8 WORKS
#clusterer = hdbscan.HDBSCAN(metric='euclidean',prediction_data=True)       # BEST is 15,12 cluster size

clusterer.fit(transformed_data)
labels = clusterer.labels_
num_clus =  np.max(clusterer.labels_)
print("Number of cluster:",num_clus+1)
print([np.count_nonzero(clusterer.labels_ == i) for i in range(-1,1+num_clus)])

clusters = [np.ma.nonzero(clusterer.labels_ == i)[0] for i in range(-1,1+num_clus)]

from sklearn.ensemble import IsolationForest
anomalies = [clusters[0]]
for cluster in clusters[1:]:
    forest = IsolationForest(random_state=314).fit_predict(transformed_data[cluster])
    anomalies.append(cluster[np.ma.nonzero(forest==-1)])


print("---Clustering done. Visualising using t-SNE---")
detected = 0
for i in range(len(data)):
    point = tags[i]    
    if tuple(point.astype('int32')) in TOI:
        if not (f := any( i in x for x in anomalies)):
            print(point.astype('int32'), clusterer.labels_[i])                                     # is a part of last line
        if f:
            detected += 1 

print(f'Data reduction: {round(100-100*sum(len(x) for x in anomalies)/(len(data)),1)}%\t Accuracy: {detected}/{len(TOI)}')


#TNSE START
#exit()
from sklearn.manifold import TSNE

def color_cluster(data,clusterer):
    global labels
    return labels

def color_TOI(data,clusterer):
    ret = []
    for i in range(len(data)):
        point = tags[i]
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

for tag,data in data_gen:
    print('aye')
    tags = np.concatenate((tags,tag))
    normed = pca.transform(StandardScaler().fit_transform(data))
    transformed_data = np.concatenate((transformed_data,normed))
    labels = np.concatenate((labels,hdbscan.prediction.approximate_predict(clusterer,normed)[0]))

data_tsne = TSNE(n_components=2).fit_transform(transformed_data)        ############# transformed or normed


ax.scatter(data_tsne[:,0], data_tsne[:, 1], s = 5, picker=5, c= labels)#color_TOI(data_tsne, clusterer))#color_cluster(transformed_data,clusterer))       # what to color? 
def onpick(event):
    ind = event.ind
    ccd_point = tags[ind][0]
    coords = (int(ccd_point[0]),int(ccd_point[1]),int(ccd_point[2]), int(ccd_point[3]))
    fig1,ax1 = plt.subplots()

    found,i = False,0
    while not found:
        try:
            lc = lcobj.LC(sectors[i],*coords)
            sec = sectors[i]
            found = True
        except OSError:
            i+=1

    print((sec ,*coords))
    #print(data_raw[ind][0][-1])
    print(labels[ind])
    ax1.scatter(lc.time,lc.flux,s=0.5)
    ax1.scatter(lc.time, lc.smooth_flux,s=0.5)


fig.canvas.mpl_connect('pick_event', onpick)
plt.show(block = False)
input()







'''




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
    lc = lcobj.LC(sector,*coords)
    print(coords)
    ax1.scatter(lc.time,lc.flux,s=0.5)



fig2.canvas.mpl_connect('pick_event', onpick)
plt.show(block = False)
input()
'''






