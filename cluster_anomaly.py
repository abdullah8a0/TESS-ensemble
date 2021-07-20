from numpy.ma import anom
from TOI_gen import TOI_list
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lcobj
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from pickle import dump,load
import matplotlib.pyplot as plt

sectors = [32]
plot_flag = False

def set_sector(sector):
    global sectors
    sectors = [sector]

def set_plot_flag(bool):
    global plot_flag
    plot_flag = bool



sector_32_must_detects = [
(1, 1, 2062, 1617), #
(1, 3, 1953, 1208),
(2, 3, 1109, 1645),
(3, 1, 1778, 738),
(3, 1, 137, 399),
(3, 1, 1032, 739),
(3, 1, 1584, 1125),
(3, 2, 100, 1095),
(3, 2, 1178, 2001),
(3, 2, 1103, 606),
(3, 3, 1933, 509),
(3, 4, 195, 1152),
(4, 1, 2045, 348),  #
(4, 2, 70, 1634),
(4, 3, 957, 1933),
(4, 3, 509, 256),
(4, 3, 1877, 1600),
]


def hdbscan_cluster(tags,data,verbose,dim,training_sector, min_size,min_samp,metric):

    # Try removing p_adner
    scaler = StandardScaler()

    data_norm = scaler.fit_transform(data)

    #### PCA START
    if verbose:
        print("---Reducing Dimensionality using PCA---")

    pca = PCA(n_components=dim)   #12,13 best, 14 prev, 15 WORKS
    pca.fit(data_norm)


    transformed_data = pca.transform(data_norm)

    #Clustering start

    import hdbscan

    if verbose:
        print("---Dimesionality Reduced. Starting Cluster using HDBSCAN---")

    if training_sector is None:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size,min_samples=min_samp ,metric=metric,prediction_data=True)       # BEST is 15,12 cluster size, 19 previous, 7 prev, 8 WORKS
        #clusterer = hdbscan.HDBSCAN(metric='euclidean',prediction_data=True)       # BEST is 15,12 cluster size
    else:
        with open(f"Pickled\\{training_sector}.p",'rb') as file:
            clusterer = load(file)


    clusterer.fit(transformed_data)

    return clusterer,transformed_data

def tsne_plot(tags,transformed_data,labels):

        from sklearn.manifold import TSNE

        plt.ion()
        fig,ax = plt.subplots()


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
            #print(data[ind][0])
            print(labels[ind])
            ax1.scatter(lc.time,lc.flux,s=0.5)
            ax1.scatter(lc.time, lc.smooth_flux,s=0.5)


        fig.canvas.mpl_connect('pick_event', onpick)
        plt.show(block = False)
        input()

def cluster_and_plot(sub_tags=None,sub_feat=None,dim =15, min_size = 8,min_samp=8, metric = 'euclidean', write=True, verbose=False, vet_clus=False, model_persistence=False, training_sector=None):
    TOI = TOI_list(sectors[0])

    data_gen = lcobj.get_sector_data(sectors,'s',verbose=False)
    tags, data = next(data_gen)
    if sub_tags is not None:
        global plot_flag
        plot_flag=True
        tags, data = sub_tags, sub_feat



#    # Try removing p_adner
#    scaler = StandardScaler()

#    data_norm = scaler.fit_transform(data)

#    #### PCA START
#    if verbose:
#        print("---Reducing Dimensionality using PCA---")

#    pca = PCA(n_components=dim)   #12,13 best, 14 prev, 15 WORKS
#    pca.fit(data_norm)


#    transformed_data = pca.transform(data_norm)

#    #Clustering start

#    import hdbscan

#    if verbose:
#        print("---Dimesionality Reduced. Starting Cluster using HDBSCAN---")

#    if training_sector is None:
#        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size,min_samples=min_samp ,metric=metric,prediction_data=True)       # BEST is 15,12 cluster size, 19 previous, 7 prev, 8 WORKS
#        #clusterer = hdbscan.HDBSCAN(metric='euclidean',prediction_data=True)       # BEST is 15,12 cluster size
#    else:
#        with open(f"Pickled\\{training_sector}.p",'rb') as file:
#            clusterer = load(file)


#    clusterer.fit(transformed_data)

    clusterer,transformed_data = hdbscan_cluster(tags,data,verbose,dim,training_sector,min_size,min_samp,metric)

    if model_persistence:
        with open(f"Pickled\\{sectors[0]}.p",'wb') as file:
            dump(clusterer,file)




    labels = clusterer.labels_
    num_clus =  np.max(clusterer.labels_)

    clus_count = [np.count_nonzero(clusterer.labels_ == i) for i in range(-1,1+num_clus)]
    
    if verbose:
        print("Number of cluster:",num_clus+1)
        print(clus_count)

    ind = np.argpartition(np.array(clus_count), -2)[-2:]


    clusters = [np.ma.nonzero(clusterer.labels_ == i)[0] for i in range(-1,1+num_clus)]

    from sklearn.ensemble import IsolationForest
    anomalies = [clusters[0]]
    
    #ind = [i for i in range(len(clusters))]
    for i in ind:
        if i==0:
            continue
        if verbose:
            print('detailing: ',i)
        cluster = clusters[i]

        predictor = IsolationForest(random_state=314)
        forest = predictor.fit_predict(transformed_data[cluster])
        anomalies.append(cluster[np.ma.nonzero(forest==-1)])

    '''         # MULTI SECTOR CODE     (Future Use)
    for tag,data in data_gen:
        print('aye')
        normed = pca.transform(StandardScaler().fit_transform(data))
        transformed_data = np.concatenate((transformed_data,normed))
        new_labels = hdbscan.prediction.approximate_predict(clusterer,normed)[0]

        new_clusters = [np.ma.nonzero(new_labels == i)[0] for i in range(-1,1+num_clus)]

        for i,cluster in enumerate(new_clusters):
            if cluster.size ==0:
                continue

            predictor = IsolationForest(random_state=314)
            forest = predictor.fit_predict(normed[cluster])

            while i>= len(anomalies):                               ################################################
                anomalies.append(np.array([]))
            
            anomalies[i] = np.append(anomalies[i],len(tags) + cluster[np.ma.nonzero(forest==-1)])

        tags = np.concatenate((tags,tag))
        
        labels = np.concatenate((labels,new_labels))
    '''

    name_ = '_'.join(str(i) for i in sectors)
    if write:
        np.savetxt(f'Results/{name_}.txt',tags[np.array([i for x in anomalies for i in x]).astype('int32')], fmt='%1d')

        ret = []
        for i,tag in enumerate(tags):
            clus = clusterer.labels_[i]
            isAnom = 1 if any(i== entry for clus in anomalies for entry in clus) else 0
            ret.append(np.array([*tag, clus,isAnom],dtype='int32'))
        
        np.savetxt(f'Processed/{name_}.txt',np.array(ret), fmt='%1d',delimiter =',')
        


    detected = 0
    must_detect = 0
    for i in range(len(tags)):
        point = tags[i]    
        if tuple(point.astype('int32')) in TOI:
            if not (f := any( i in x for x in anomalies)):
                if verbose:
                    print(point.astype('int32'), clusterer.labels_[i])                                     # is a part of last line
            if f:
                detected += 1 
        if tuple(point.astype('int32')) in sector_32_must_detects:
            if not (f := any( i in x for x in anomalies)):
                print('Must detect: ',point.astype('int32'), clusterer.labels_[i])                                     # is a part of last line
            if f:
                must_detect += 1 


    print(f'Data reduction: {round(100-100*sum(len(x) for x in anomalies)/(len(tags)),1)}%\t Accuracy: {detected}/{len(TOI)}\nCompulsory Accuracy in Sector 32: {must_detect}/{len(sector_32_must_detects)}')

    if plot_flag:
            
        if verbose:
            print("---Clustering done. Visualising using t-SNE---")
        #TNSE START


        tsne_plot(tags,transformed_data,labels)


#        from sklearn.manifold import TSNE

#        def color_cluster(data,clusterer):
#            return labels

#        def color_TOI(data,clusterer):
#            ret = []
#            for i in range(len(data)):
#                point = tags[i]
#                if tuple(point.astype('int32')) in TOI:
#                    ret.append('red')
#                    #print(point.astype('int32').tolist())
#                else:
#                    ret.append('green')
#            return np.array(ret)


#        #def color_anomaly(data):
#        #    return forest


#        plt.ion()
#        fig,ax = plt.subplots()


#        data_tsne = TSNE(n_components=2).fit_transform(transformed_data)        ############# transformed or normed


#        ax.scatter(data_tsne[:,0], data_tsne[:, 1], s = 5, picker=5, c= labels)#color_TOI(data_tsne, clusterer))#color_cluster(transformed_data,clusterer))       # what to color? 
#        def onpick(event):
#            ind = event.ind
#            ccd_point = tags[ind][0]
#            coords = (int(ccd_point[0]),int(ccd_point[1]),int(ccd_point[2]), int(ccd_point[3]))
#            fig1,ax1 = plt.subplots()

#            found,i = False,0
#            while not found:
#                try:
#                    lc = lcobj.LC(sectors[i],*coords)
#                    sec = sectors[i]
#                    found = True
#                except OSError:
#                    i+=1

#            print((sec ,*coords))
#            #print(data[ind][0])
#            print(labels[ind])
#            ax1.scatter(lc.time,lc.flux,s=0.5)
#            ax1.scatter(lc.time, lc.smooth_flux,s=0.5)


#        fig.canvas.mpl_connect('pick_event', onpick)
#        plt.show(block = False)
#        input()

    if vet_clus:
        import cluster_vetter
        for i in range(num_clus+2):
            print(f"-- Showing cluster {i-1} --")
            cluster_vetter.vet_clusters(sectors[0],tags,transformed_data,clusters[i])









