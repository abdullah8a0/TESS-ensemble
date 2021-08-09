from TOI_gen import TOI_list
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import lcobj
import numpy as np
from pickle import dump,load
import matplotlib.pyplot as plt

sector = 32
plot_flag = False

def set_sector(sec):
    global sector
    sector = sec

def set_plot_flag(bool):
    global plot_flag
    plot_flag = bool



def score(tags,sec):
    if tags is None:
        tags = (np.genfromtxt(f'Results\\{sec}.txt',delimiter=',')[:,:4]).astype('int32')

    must_detects = TOI_list(sec)
    
    temp = np.array(must_detects[:])
    seen =[]
    if tags.shape[1] ==5:
        assert tags[0][0] == sec
        tags = tags[:,1:]
    count = 0 
    for i,tag in enumerate(tags):
        if (t := tuple(tag.astype('int32'))) in must_detects:
            count +=1
            seen.append(must_detects.index(t))
    temp = np.delete(temp,seen,0)
    if temp.size != 0:
        for k in temp:
            print("Must detect: ",k)
            #lcobj.LC(sec,*k).plot()
    return count    



def scale_simplify(data,verbose,dim):

    scaler = StandardScaler()

    data_norm = scaler.fit_transform(data)

    if verbose:
        print("---Reducing Dimensionality using PCA---")

    #reducer = umap.UMAP(n_neighbors=10,n_components=dim,min_dist=0.01,random_state=314)
    
    
    pca = PCA(n_components=dim) 
    pca.fit(data_norm)
    
    s = sum(pca.explained_variance_ratio_)
    if verbose:
        print(s:=sum(pca.explained_variance_ratio_))
    assert s>0.85 # if fails then increase dimensions

    transformed_data = pca.transform(data_norm)
    #transformed_data = reducer.fit_transform(data_norm)
    return transformed_data

def hdbscan_cluster(transformed_data,verbose,training_sector, min_size,min_samp,metric,type = 'eom',epsilon=0):


    #Clustering start

    import hdbscan

    if verbose:
        print("---Dimesionality Reduced. Starting Cluster using HDBSCAN---")

    if training_sector is None:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size,cluster_selection_epsilon=epsilon ,min_samples=min_samp,metric=metric,prediction_data=True,cluster_selection_method=type)       # BEST is 15,12 cluster size, 19 previous, 7 prev, 8 WORKS
        clusterer.fit(transformed_data)
        return clusterer,clusterer.labels_
    else:
        with open(f"Pickled\\{training_sector}.p",'rb') as file:
            clusterer = load(file)
        return clusterer,hdbscan.approximate_predict(clusterer,transformed_data)[0]


def umap_plot(tags,transformed_data,labels,TOI=None,normalized=True):
    tags_find = lcobj.TagFinder(tags)
    TOI_ind = []
    TOI = TOI if TOI is not None else []
    for tag in TOI:
        try:
            TOI_ind.append(tags_find.find(tag))
        except Exception:
            continue

    reducer = umap.UMAP(n_neighbors=30,min_dist=0.01,random_state=314)

    data_umap = reducer.fit_transform(transformed_data)
    plt.ion()
    fig,ax = plt.subplots()
    if not TOI:
        ax.scatter(data_umap[:,0], data_umap[:, 1], s = 5, picker=5, c= labels)
    else:
        col = np.where(([True if i not in TOI_ind else False for i in range(len(labels))]),labels,[0.5]*len(labels))
        ax.scatter(data_umap[:,0], data_umap[:, 1], s = 5, picker=5, c= col) 
    def onpick(event):
        ind = event.ind
        ccd_point = tags[ind][0]
        coords = (int(ccd_point[0]),int(ccd_point[1]),int(ccd_point[2]), int(ccd_point[3]))
        fig1,ax1 = plt.subplots()

        print((sector ,*coords))
        lc = lcobj.LC(sector,*coords)
        sec = sector
    
        granularity = 1.0           # In days
        bins = granularity*np.arange(27)
        bin_map = np.digitize(lc.time-lc.time[0], bins)

        feat = []
        for bin in bins:
            dp_in_bin = np.ma.nonzero(bin_map == bin+1)     
            flux, time = lc.normed_flux[dp_in_bin], lc.time[dp_in_bin]
            if flux.size in {0,1}:
                slope ,c, r = 0,0,1
            else:
                slope, c, r = stats.linregress(time,flux)[:3]  #(slope,c,r)
        
            feat.append([slope, c,r**2])
        fit = []
        time = [ [t for t in lc.time if lc.time[0] + day <= t < lc.time[0]+day+1] for day in range(27)]
        for day in range(27):
            m,c,_ = feat[day]
            fit += [m*t+c for t in time[day]]
        print(labels[ind])
        if normalized:
            ax1.scatter(lc.time,lc.normed_flux,s=0.5)
        else:
            ax1.scatter(lc.time,lc.flux,s=0.5)
        ax1.scatter(lc.time,fit,s=0.5)


    fig.canvas.mpl_connect('pick_event', onpick)
    input('Press Enter to continue\n')
    plt.show(block=False)

from sklearn.manifold import TSNE
def tsne_plot(tags,transformed_data,labels,TOI=None,normalized=True,with_sec=False):
    tags_find = lcobj.TagFinder(tags)
    TOI_ind = []
    TOI = TOI if TOI is not None else []
    for tag in TOI:
        try:
            TOI_ind.append(tags_find.find(tag))
        except Exception:
            continue

    plt.ion()
    fig,ax = plt.subplots()
    data_tsne = TSNE(n_components=2).fit_transform(transformed_data)        ############# transformed or normed
    if not TOI:
        ax.scatter(data_tsne[:,0], data_tsne[:, 1], s = 5, picker=5, c= labels)
    else:
        col = np.where(([True if i not in TOI_ind else False for i in range(len(labels))]),labels,[0.5]*len(labels))
        ax.scatter(data_tsne[:,0], data_tsne[:, 1], s = 5, picker=5, c= col) 
    
    def onpick(event):
        ind = event.ind
        ccd_point = tags[ind][0]
        if not with_sec:
            coords = (int(ccd_point[0]),int(ccd_point[1]),int(ccd_point[2]), int(ccd_point[3]))
            print((sector ,*coords))
            lc = lcobj.LC(sector,*coords)
        else:
            coords = (int(ccd_point[0]),int(ccd_point[1]),int(ccd_point[2]), int(ccd_point[3]),int(ccd_point[4]))
            print(coords)
            lc = lcobj.LC(*coords)
        fig1,ax1 = plt.subplots()
        sec = sector
    
        granularity = 1.0           # In days
        bins = granularity*np.arange(27)
        bin_map = np.digitize(lc.time-lc.time[0], bins)

        feat = []
        for bin in bins:
            dp_in_bin = np.ma.nonzero(bin_map == bin+1)     
            flux, time = lc.normed_flux[dp_in_bin], lc.time[dp_in_bin]
            if flux.size in {0,1}:
                slope ,c, r = 0,0,1
            else:
                slope, c, r = stats.linregress(time,flux)[:3]  #(slope,c,r)
        
            feat.append([slope, c,r**2])
        fit = []
        time = [ [t for t in lc.time if lc.time[0] + day <= t < lc.time[0]+day+1] for day in range(27)]
        for day in range(27):
            m,c,_ = feat[day]
            fit += [m*t+c for t in time[day]]
        print(labels[ind])
        if normalized:
            ax1.scatter(lc.time,lc.normed_flux,s=0.5)
        else:
            ax1.scatter(lc.time,lc.flux,s=0.5)
        ax1.scatter(lc.time,fit,s=0.5)


    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show(block = False)
    input('Press Enter to continue\n')

def cluster_and_plot(sub_tags=None,sub_feat=None,show_TOI=True,dim =15, min_size = 8,min_samp=8,type='eom', metric = 'euclidean', write=True, verbose=False, vet_clus=False, model_persistence=False, training_sector=None):
    TOI = TOI_list(sector) if show_TOI else None

    data_gen = lcobj.get_sector_data(sector,'s',verbose=False)
    tags, data = next(data_gen)
    if sub_tags is not None:
        global plot_flag
        plot_flag=True
        tags, data = sub_tags, sub_feat

    transformed_data = scale_simplify(data,verbose,dim)

    clusterer,new_labels = hdbscan_cluster(transformed_data,verbose,training_sector,min_size,min_samp,metric,type=type)

    if model_persistence:
        with open(f"Pickled\\{sector}.p",'wb') as file:
            dump(clusterer,file)




    labels = new_labels
    num_clus =  np.max(new_labels)

    clus_count = [np.count_nonzero(new_labels == i) for i in range(-1,1+num_clus)]
    
    if verbose:
        print("Number of cluster:",num_clus+1)
        print(clus_count)

    ind = np.argpartition(np.array(clus_count), -2)[-2:]


    clusters = [np.ma.nonzero(new_labels == i)[0] for i in range(-1,1+num_clus)]

    from sklearn.ensemble import IsolationForest
    anomalies = [clusters[0]]
    
    #ind = [i for i in range(len(clusters))]
    for i in ind:
        if i==0:
            continue
        if verbose:
            print('detailing: ',i)
        cluster = clusters[i]

        predictor = IsolationForest(random_state=314,contamination=0.1)
        forest = predictor.fit_predict(transformed_data[cluster])
        anomalies.append(cluster[np.ma.nonzero(forest==-1)])


    name_ = str(sector)
    if write:
        np.savetxt(f'Results/{name_}.txt',tags[np.array([i for x in anomalies for i in x]).astype('int32')], fmt='%1d')

        ret = []
        for i,tag in enumerate(tags):
            clus = new_labels[i]
            isAnom = 1 if any(i== entry for clus in anomalies for entry in clus) else 0
            ret.append(np.array([*tag, clus,isAnom],dtype='int32'))
        
        np.savetxt(f'Processed/{name_}.txt',np.array(ret), fmt='%1d',delimiter =',')
        


    must_detect = score(np.array([tags[i,:] for x in anomalies for i in x]),sector)
    for i in range(len(tags)):
        point = tags[i]    
        if tuple(point.astype('int32')) in TOI:
            if not (f := any( i in x for x in anomalies)):
                if verbose:
                    print(point.astype('int32'), new_labels[i])                                     # is a part of last line
#            if f:
#                detected += 1 
#           if tuple(point.astype('int32')) in sector_32_must_detects:
#               if not (f := any( i in x for x in anomalies)):
#                   print('Must detect: ',point.astype('int32'), new_labels[i])                                     # is a part of last line
#               if f:
#                   must_detect += 1 


    print(f'Data reduction: {round(100-100*sum(len(x) for x in anomalies)/(len(tags)),1)}%\t Score: {must_detect}/{len(TOI_list(sector))}')

    if plot_flag:
            
        if verbose:
            print("---Clustering done. Visualising using t-SNE---")
        #TNSE START
        import time
        start = time.time()
        umap_plot(tags,transformed_data,labels,TOI=TOI)
        mid = time.time()
        #tsne_plot(tags,transformed_data,labels,TOI=None)
        end = time.time()
        #print(f'UMAP: {mid-start}s      TSNE: {end-mid}s')


    if vet_clus:
        import cluster_vetter
        for i in range(num_clus+2):
            print(f"-- Showing cluster {i-1} --")
            cluster_vetter.vet_clusters(sector,tags,transformed_data,clusters[i])









