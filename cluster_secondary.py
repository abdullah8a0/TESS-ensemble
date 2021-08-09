from matplotlib import pyplot as plt
import numpy as np
import sklearn
import lcobj
import concurrent.futures
from scipy import stats
import cluster_anomaly
from sklearn.mixture import GaussianMixture



def w_metric(vec1,vec2):
    m1,r1 = vec1[::2],vec1[1::2]
    m2,r2 = vec2[::2],vec2[1::2]
    m1 = np.arctan(m1)
    m2 = np.arctan(m2)
    return sum(np.sqrt(np.abs(r1-r2)*(m1-m2)**2))



def func(tag):
    lc = lcobj.LC(*tag).remove_outliers()
    delta = lc.flux -lc.smooth_flux
    return stats.anderson(delta)[0]


def cluster_secondary(sector,verbose=False):
    raw_data = np.genfromtxt(f'Results\\{sector}.txt',delimiter =',')
    tags ,feat_v, feat_s = raw_data[:,:4].astype('int32'),raw_data[:,4:-30],raw_data[:,-30:]
    feat_s = np.delete(feat_s,[],1)
    tags = np.concatenate((sector*np.ones(len(tags)).reshape(len(tags),1),tags),axis=1).astype('int32')

    #remove feat_s


    with concurrent.futures.ProcessPoolExecutor() as executer:
        results = executer.map(func,tags)
        data = []
        for i,feat in enumerate(results):
            if i%10 == 0:
                print(i)
            data.append(feat)
    feat_s = np.concatenate((feat_s,np.array(data).reshape(len(data),1)),axis=1)
    feat_s = feat_s[:,(-1,-3)]

    #feat_v = feat_v[:,::2]# removing r**2

    #cluster_anomaly.cluster_and_plot(sub_tags= tags,sub_feat=feat_v,dim=54 \
    #    ,min_size=5,min_samp=1,write=False,verbose=True,metric=w_metric,type='eom')   # 35 for feat_v


    #implement distance net


    transformed_data = cluster_anomaly.scale_simplify(feat_s,verbose,feat_s.shape[1])
    #dist = np.array([[w_metric(vec1,vec2) for vec2 in transformed_data] for vec1 in transformed_data])
    clusterer,labels = cluster_anomaly.hdbscan_cluster(transformed_data,verbose,None,5,5,'euclidean')
    cluster_anomaly.tsne_plot(tags,transformed_data,labels,with_sec=True)
#    for i in range(2,100):
#        clusterer = sklearn.cluster.DBSCAN(eps=i,metric='precomputed').fit(dist)
#        labels = clusterer.labels_

#        cluster_anomaly.tsne_plot(tags,transformed_data,labels)





    



if __name__ == '__main__':
    sector =37
    cluster_secondary(sector,verbose=True)

