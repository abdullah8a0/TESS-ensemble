from matplotlib import pyplot as plt
import numpy as np
import sklearn
from accuracy_model import Data
import lcobj
import concurrent.futures
from scipy import stats
import cluster_anomaly
from sklearn.mixture import GaussianMixture
from pathlib import Path


def w_metric(vec1,vec2):
    m1,r1 = vec1[::2],vec1[1::2]
    m2,r2 = vec2[::2],vec2[1::2]
    m1 = np.arctan(m1)
    m2 = np.arctan(m2)
    return sum(np.sqrt(np.abs(r1-r2)*(m1-m2)**2))

def sim_score(sig1, sig2):
    return max(sum(sig1),sum(sig2)) -  np.max(np.convolve(sig1,sig2))



def func(tag):
    lc = lcobj.LC(*tag).remove_outliers()
    granularity = 1.0/3           # In days
    bins = granularity*np.arange(round(27/granularity))
    #print(bins)
    bin_map = np.digitize(lc.time-lc.time[0], bins)

    signature = []
    total_d = 0
    for bin in bins:#range(1,np.max(bin_map)+1):
        dp_in_bin = np.ma.nonzero(bin_map == round(bin/granularity)+1)
        flux, time = lc.flux[dp_in_bin], lc.time[dp_in_bin]
        _, ind = np.unique(time, return_index=True)
        flux, time = flux[ind], time[ind]

        #print(bin,flux)
        #input()
        if flux.size == 0:
            signature.append(0)
            continue
        if np.mean(flux) > lc.std + lc.mean:
            signature.append(1)
        else:
            signature.append(0)

        total_d +=1
    #time = [ [t for t in lc.time if lc.time[0] + hr8*granularity <= t < lc.time[0]+(hr8+1)*granularity] for hr8 in range(round(27/granularity))]
    #fit = []
    #for day in range(round(27/granularity)):
    #    i = interesting_d[day]
    #    fit += [i for t in time[day]]


    return [*tag,signature]



def cluster_secondary_run(sector,verbose=False):
    """
    Uhhh WTF is this? I dont remember coding this. TODO: Figure out what this does??
    """

    raw_data = np.genfromtxt(Path(f'Results/{sector}.txt'),delimiter =',')
    tags,feat_s = raw_data[:,:4].astype('int32'),raw_data[:,30:]
    #tags,feat_s = next(lcobj.get_sector_data(sector,'s',verbose=verbose))
    #feat_s = np.delete(feat_s,[],1)
    tags = np.concatenate((sector*np.ones(len(tags)).reshape(len(tags),1),tags),axis=1).astype('int32')
    with concurrent.futures.ProcessPoolExecutor() as executer:
        results = executer.map(func,tags)
        data = []
        for i,feat in enumerate(results):
            if i%10 == 0:
                print(i)
            if feat is not None:
                data.append(feat)

    feat_s = np.array(data)

    transformed_data = cluster_anomaly.scale_simplify(feat_s,verbose,feat_s.shape[1])
    metric = 'euclidean'#'precomputed'
    clusterer,labels = cluster_anomaly.hdbscan_cluster(transformed_data,verbose,None,5,5,metric)
    cluster_anomaly.tsne_plot(tags,transformed_data,labels,with_sec=True)


def forwarding(tags,data_api: Data = None,verbose=False):
    datafinder = data_api
    tags_without_sector = tags
    #tagsfinder = lcobj.TagFinder(tags)
    #sector = datafinder.sector
    #tags = np.concatenate((sector*np.ones(len(tags)).reshape(len(tags),1),tags),axis=1).astype('int32')

    #with concurrent.futures.ProcessPoolExecutor() as executer:
    #    results = executer.map(func,tags)
    #    data = []
    #    for i,feat in enumerate(results):
    #        if i%10 == 0:
    #            print(i)
    #        data.append(feat)
    
    #data = np.array(data).astype('int32')
    #shuffled_tags,signat = data[:,:5],data[:,5:]
    data = datafinder.get_some(tags=tags,type = 'signat')
    longs = np.zeros(data.shape[0]).reshape(-1,1)
    short = np.zeros(data.shape[0]).reshape(-1,1)
    cons = np.zeros(data.shape[0]).reshape(-1,1)
    for i in range(data.shape[1]):
        slice = data[:,i].reshape(-1,1)
        #print(f'{longs} + {slice} = {longs + slice}')

        cons = np.where(slice==0,slice,cons+slice)
        short = np.where(cons==1,short + cons,short)
        longs = np.where(cons==3,longs+slice,longs)

    #print(longs)
    #print(short)
    short = (short - longs).reshape(1,-1)[0]
    longs = longs.reshape(1,-1)[0]

    to_forward = np.logical_and(np.logical_and(longs<3,longs>0),short<2)
    #include iso forest 
    return np.array([tags_without_sector[forwarded_ind] for forwarded_ind in np.ma.nonzero(to_forward)[0]])




    



if __name__ == '__main__':
    sector = 37
    cluster_secondary_run(sector,verbose=True)

