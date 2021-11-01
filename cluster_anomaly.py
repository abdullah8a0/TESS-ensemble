from accuracy_model import Data
import cluster_secondary 
import umap
import hdbscan
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import lcobj
import numpy as np
from pickle import dump,load
from pathlib import Path
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


def scale_simplify(data,verbose,dim):

    scaler = StandardScaler()

    data_norm = scaler.fit_transform(data)

    if verbose:
        print("---Reducing Dimensionality using PCA---")

    
    pca = PCA(n_components=dim) 
    pca.fit(data_norm)
    
    s = sum(pca.explained_variance_ratio_)
    if verbose:
        print(s:=sum(pca.explained_variance_ratio_))
    assert s>0.85 # if fails then increase dimensions

    transformed_data = pca.transform(data_norm)
    return transformed_data

def vet_clusters(sector, tags, feat_data, clus_ind,processed = None):
    processed = processed if processed is not None else np.genfromtxt(f'Processed\\{sector}.txt', delimiter=',').astype('int32')


    pro_tags, pro_clus, labels = processed[:,:4][clus_ind],processed[:,4][clus_ind],-processed[:,5][clus_ind]

    #assert (pro_tags == tags[clus_ind]).all()
    tags = tags[clus_ind]

    labels = np.where(labels ==-1, labels,pro_clus)

    transformed_data = feat_data[clus_ind]

    tsne_plot(sector,tags,transformed_data,labels)
    
def hdbscan_cluster(transformed_data,training_sector, min_size,min_samp,metric,epsilon=0):


    #Clustering start



    if training_sector is None:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size,cluster_selection_epsilon=epsilon ,min_samples=min_samp,metric=metric,prediction_data=True)       # BEST is 15,12 cluster size, 19 previous, 7 prev, 8 WORKS
        clusterer.fit(transformed_data)
        return clusterer,clusterer.labels_
    else:
        with open(Path(f"Pickled/{training_sector}.p"),'rb') as file:
            clusterer = load(file)
        return clusterer,hdbscan.approximate_predict(clusterer,transformed_data)[0]


def umap_plot(tags,transformed_data,labels,TOI=None,normalized=True,with_sec=False):
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
        print(labels[ind])
        fig1,ax1 = plt.subplots()

        if not with_sec:
            coords = (int(ccd_point[0]),int(ccd_point[1]),int(ccd_point[2]), int(ccd_point[3]))
        #    print((sector ,*coords))
        #    lc = lcobj.LC(sector,*coords)
        else:
            coords = (int(ccd_point[0]),int(ccd_point[1]),int(ccd_point[2]), int(ccd_point[3]),int(ccd_point[4]))
            print(coords)
            lc = lcobj.LC(*coords)
        lc.remove_outliers().plot()

    fig.canvas.mpl_connect('pick_event', onpick)
    input('Press Enter to continue\n')
    plt.show(block=False)

def tsne_plot(sector,tags,transformed_data,labels,normalized=True,with_sec=False):

    fig,ax = plt.subplots()
    data_tsne = TSNE(n_components=2).fit_transform(transformed_data)        ############# transformed or normed
    ax.scatter(data_tsne[:,0], data_tsne[:, 1], s = 5, picker=5, c= labels) 
    
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
        fig,ax1 = plt.subplots()
    
        print(labels[ind])
        if normalized:
            ax1.scatter(lc.time,lc.normed_flux,s=0.5)
        else:
            ax1.scatter(lc.time,lc.flux,s=0.5)
        fig.show()

    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()
    #input('Press Enter to continue\n')

def cluster_and_plot(tags,datafinder : Data = None,plot_flag = False,dim =15, metric = 'euclidean', write=True, verbose=False, vet_clus=False, model_persistence=False, training_sector=None):

    tag_finder = lcobj.TagFinder(tags)
    data = datafinder.get_all(type='scalar')

    transformed_data = scale_simplify(data,verbose,dim)
    if verbose:
        print("---Dimesionality Reduced. Starting Cluster using HDBSCAN---")
    
    size_base,samp_base = 15,3
    
    br = False
    
    while not br:
        for size,samp in [(i,j) for i in range(size_base-3,size_base+3) for j in range(samp_base-3,samp_base+3) if i > 0 and j>0]:
            clusterer,new_labels = hdbscan_cluster(transformed_data,training_sector,size,samp,metric)


            ## ONLY FOR SCORING
            labels = new_labels
            num_clus =  np.max(new_labels)
            clus_count = [np.count_nonzero(new_labels == i) for i in range(-1,1+num_clus)]
            if len(clus_count) == 1:
                print(f'{size}, {samp}\t: No Clustering\n')
                continue
            ind = np.argpartition(np.array(clus_count), -2)[-2:]
            for i in ind:
                if i==0:
                    continue
                #if verbose:
                    #print('detailing: ',i)
                clusters = [np.ma.nonzero(new_labels == i)[0] for i in range(-1,1+num_clus)]

                anomalies = [clusters[0]]
                cluster = clusters[i]

                predictor = IsolationForest(random_state=314,contamination=0.1)
                forest = predictor.fit_predict(transformed_data[cluster])
                anomalies.append(cluster[np.ma.nonzero(forest==-1)])
            
            reduction = 1-sum(len(i) for i in anomalies)/tags.shape[0]
            a,b ='*','.'
            print(f'{size}, {samp}\t: {a if (good:=(0.65>reduction>0.5)) else b}\t{reduction}\n\t  {clus_count}')
            if good:
                br = True
                break


        if not br:
            lis =  [int(i) for i in input('Enter new center or choose from above: ').split(' ')]
            size_base, samp_base = lis[0],lis[1]
            if len(lis) == 3:
                size,samp = size_base,samp_base
                br = True    

    sel_size,sel_samp = size,samp

    clusterer,new_labels = hdbscan_cluster(transformed_data,training_sector,int(sel_size),int(sel_samp),metric)

    if plot_flag:
            
        if verbose:
            print("---Clustering done. Visualising using t-SNE---")
        tsne_plot(datafinder.sector,tags,transformed_data,new_labels)

    num_clus =  np.max(new_labels)

    clus_count = [np.count_nonzero(new_labels == i) for i in range(-1,1+num_clus)]
    


    if verbose:
        print("Number of cluster:",num_clus+1)
        print(clus_count)

    #if model_persistence:
    #    with open(Path(f"Pickled/{sector}.p"),'wb') as file:
    #        dump(clusterer,file)


    clusters = [np.ma.nonzero(new_labels == i)[0] for i in range(-1,1+num_clus)]

    anomalies = [clusters[0]]
    
    ### FORWARD BYPASS
    main_blob = tags[(main_blob_ind:=clusters[clus_count.index(max(clus_count))])]
    forwards = cluster_secondary.forwarding(main_blob,datafinder)
    forwards_ind = np.array([tag_finder.find(tag) for tag in forwards])
    forwards_data = transformed_data[forwards_ind]

    forest = IsolationForest(random_state=314,contamination=0.1,n_jobs=-1)
    forest.fit(transformed_data[main_blob_ind])
    score_ = forest.decision_function(forwards_data)
    #print(score_)
    good_to_forward_ind = np.argpartition(-score_, -len(score_)//10)[-len(score_)//10:]
    #print(good_to_forward_ind)

    anomalies.append(forwards_ind[good_to_forward_ind])

    ### END FORWARDING

    #name_ = str(sector)
    #if write:
    #    np.savetxt(Path(f'Results/{name_}.txt'),tags[np.array([i for x in anomalies for i in x]).astype('int32')], fmt='%1d')

    #    ret = []
    #    for i,tag in enumerate(tags):
    #        clus = new_labels[i]
    #        isAnom = 1 if any(i== entry for clus in anomalies for entry in clus) else 0
    #        ret.append(np.array([*tag, clus,isAnom],dtype='int32'))
    #    
    #    np.savetxt(Path(f'Processed/{name_}.txt'),np.array(ret), fmt='%1d',delimiter =',')
        
    print(f'Data reduction: {round(100-100*sum(len(x) for x in anomalies)/(len(tags)),1)}%')


#    if vet_clus:
#        for i in range(num_clus+2):
#            print(f"-- Showing cluster {i-1} --")
#            vet_clusters(sector,tags,transformed_data,clusters[i])

    return tags[np.array([i for x in anomalies for i in x]).astype('int32')]







