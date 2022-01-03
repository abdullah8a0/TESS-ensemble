#import matplotlib.pyplot as plt
from accuracy_model import Data,transient_tags
from lcobj import LC
from cluster_anomaly import hdbscan_cluster, scale_simplify,tsne_plot,umap_plot
import numpy as np
import concurrent.futures

WIDTH, HEIGHT = 30, 20

def bin(tag)->np.ndarray:
    im = np.zeros((HEIGHT,WIDTH),dtype='int32')
    lc = LC(*tag).remove_outliers()
    for x,y in zip(lc.normed_time,lc.normed_flux):
        dx, dy = int(x*WIDTH) if x!=1 else WIDTH-1,int(y*HEIGHT) if y!=1 else HEIGHT-1
        im[19-dy,dx] +=1
    #im = plt.imshow(im, cmap='hot', aspect=1) 
    #plt.show()
    return im

def classify(tags,api):
    data = []
    with concurrent.futures.ProcessPoolExecutor() as executer:
        results = executer.map(bin,tags)
        Data = []
        for i,feat in enumerate(results):
            if (i%100 == 0):
                print(i)
            if feat is not None and np.all(np.isfinite(feat)) and not np.any(np.isnan(feat)):
                Data.append(feat.flatten())
        Data = np.array(Data)
        data = Data if data == [] else np.concatenate((data,Data))
    print(data.shape)
    normed = scale_simplify(data,False,10,skip=True)


    transformed_data = normed
    training_sector = None
    print(training_sector)
    metric = 'euclidean'
    size_base,samp_base = 12,1
    size,samp = size_base,samp_base
    HIGH,LOW = 0.72,0.55
    br = True
    
    while not br:
        for size,samp in [(i,j) for i in range(size_base-3,size_base+3) for j in range(samp_base-3,samp_base+3) if i > 0 and j>0]:
            clusterer,new_labels = hdbscan_cluster(transformed_data,training_sector,size,samp,metric)


            ## ONLY FOR SCORING
            num_clus =  np.max(new_labels)
            clus_count = [np.count_nonzero(new_labels == i) for i in range(-1,1+num_clus)]
            if len(clus_count) == 1:
                print(f'{size}, {samp}\t: No Clustering\n')
                continue
            
            print(f'{size}, {samp}\t: \t \n\t  {clus_count}')


        if not br:
            lis =  [int(i) for i in input('Enter new center or choose from above: ').split(' ')]
            size_base, samp_base = lis[0],lis[1]
            if len(lis) == 3:
                size,samp = size_base,samp_base
                br = True    

    sel_size,sel_samp = size,samp

    clusterer,new_labels = hdbscan_cluster(transformed_data,training_sector,int(sel_size),int(sel_samp),metric)



    umap_plot(tags[0,0],tags,normed,new_labels,with_sec=True,TOI=api)

if __name__ == '__main__':
    sector = 32
    data = Data(sector,'s')
    data.new_insert([i for i in range(50)])
    tags = np.concatenate((data.stags[:,:],[transient_tags[i] for i in data.ind[-1]]))
    l = len(tags)
    tags_with_sector = np.concatenate((np.array([sector]*l).reshape(l,1),np.array(tags)),axis=1)
    classify(tags_with_sector,data)
    pass
