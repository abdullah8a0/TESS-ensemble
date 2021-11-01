from TOI_gen import TOI_list
import numpy as np
from lcobj import LC, get_sector_data, set_base
import matplotlib
#matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from scipy import stats
import lcobj
import concurrent.futures
from sklearn.mixture import GaussianMixture
from sklearn.metrics import r2_score

#(3, 4, 805, 1076)
def func(tag):
    lc = lcobj.LC(43,*tag)
    
    step_init = lc.normed_smooth_flux[0]
    step_fin = lc.normed_smooth_flux[-1]
    center = np.argmax(np.abs(lc.normed_smooth_flux[1:] - lc.normed_smooth_flux[:-1]))
    k = len(lc.normed_smooth_flux)
    center = center if k> center else k
    center = center if center > 0 else 0

    step_center = (step_fin + step_init)/2
    band_width = np.min(np.abs(lc.normed_flux - step_center))



    fit = np.concatenate((np.array([step_init]*center), np.array([step_fin]*(k-center))))
    
    perr = r2_score(lc.normed_smooth_flux,fit)
    
    return np.log(1/(1-perr))
    
    delta = lc.flux -lc.smooth_flux
    return stats.anderson(delta)[0]


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
        if flux.size <= 5:
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


    return signature

def get_feat(tags):

    with concurrent.futures.ProcessPoolExecutor() as executer:
        results = executer.map(func,tags)
        data = []
        for i,feat in enumerate(results):
            if i%10 == 0:
                print(i)
            data.append(feat)
    return data     #

    data = np.array(data)
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


    feat = np.logical_and(np.logical_and(longs<3,longs>0),short<2) 
    return feat

def get_iso_feat(sector,tags,data):
    from sklearn.ensemble import IsolationForest
    from cluster_anomaly import scale_simplify,hdbscan_cluster

    transformed_data = scale_simplify(data,True,15)
    clusterer,labels = hdbscan_cluster(transformed_data,False,None,14,3,'euclidean')

    num_clus =  np.max(labels)

    clus_count = [np.count_nonzero(labels == i) for i in range(-1,1+num_clus)]
    largest_clus = clus_count.index(max(clus_count))
    

    clusters = [np.ma.nonzero(labels == i)[0] for i in range(-1,1+num_clus)]

    ind = clusters[largest_clus]

    new_tags= tags[ind,:]
    transformed_data = transformed_data[ind,:]

    iso = IsolationForest(n_jobs=-1,random_state=314)
    anomalous = iso.fit_predict(transformed_data)
    score = iso.decision_function(transformed_data)
    return score,new_tags

from pathlib import Path
def plot_feat(sector):
    #tags, data = next(get_sector_data(sector, 's',verbose=False))

    file_path = Path(f'Results/{sector}.txt')
    raw_data = np.genfromtxt(file_path,delimiter = ',')
    tags,data = raw_data[:,:4].astype('int32'), raw_data[:,4:]
    #file_path= f'Results\\{sector}.txt'
    #tags = np.genfromtxt(file_path,delimiter = ',')[:,:4].astype('int32')

    #tags = tags[:100,:]

    def TOI_color(data):
        ret = []
        for i in range(len(data)):
            point = tags[i]
            if tuple(point.astype('int32').tolist()) in TOI:
                ret.append('red')
            else:
                ret.append('green')
        return np.array(ret)

    def TOI_and_sign_color(data):
        ret = []
        import cluster_secondary
        for i in range(len(data)):
            longs,short,cons = 0,0,0
            point = tags[i]
            tag=tuple(point.astype('int32').tolist())
            signature = cluster_secondary.func(tuple([sector,*tag]))
            for i in range(len(signature)):
                slice = signature[i]
                if slice ==0:
                    cons = 0
                else:
                    cons +=1
                if cons == 1:
                    short +=1
                if cons == 3:
                    longs +=1

            short = (short - longs)
            to_forward = longs<3 and longs>0 and short<2
            if tag in TOI:
                ret.append('red')
            elif to_forward:
                ret.append('blue')
            else:
                ret.append('green')
        return np.array(ret)

    '''
    feat = np.array([amp,better_amp,med,mean,std,slope,r,skew,max_slope,\
    beyond1std, delta_quartiles, flux_mid_20, *flux_mid_35*, flux_mid_50, \
    flux_mid_65, flux_mid_80, *cons*, slope_trend, var_ind(near 2 means normal dist), med_abs_dev, \
    H1, R21, R31, Rcs, l , med_buffer_ran, perr,band_width, StetK, p_ander, day_of_i])
    '''

    feat = get_feat(tags)
    #feat,tags = get_iso_feat(sector,tags,data)

    #plt.ion()

    fig,ax = plt.subplots()
    ax.scatter(range(len(feat)),feat,s=0.5,picker=5,c=TOI_color(feat))
    def onpick(event):
        ind = event.ind
        ccd_point = tags[ind][0]
        coords = (int(ccd_point[0]),int(ccd_point[1]),int(ccd_point[2]), int(ccd_point[3]))
        fig,ax1 = plt.subplots()
        lc = lcobj.LC(sector,*coords)
        lc.remove_outliers()
        ax1.scatter(lc.time,lc.flux,s=0.5)
        print(coords)
        ax1.scatter(lc.time,lc.smooth_flux,s=0.5)

        step_init = lc.smooth_flux[0]
        step_fin = lc.smooth_flux[-1]
        center = np.argmax(np.abs(lc.smooth_flux[1:] - lc.smooth_flux[:-1]))
        k = len(lc.smooth_flux)
        center = center if k> center else k
        center = center if center > 0 else 0

        step_center = (step_fin + step_init)/2
        band_width = np.min(np.abs(lc.flux - step_center))



        fit = np.concatenate((np.array([step_init]*center), np.array([step_fin]*(k-center))))

        ax1.scatter(lc.time, fit,s=0.5)
        #ax1.scatter(lc.time, lc.N*[lc.mean+lc.std],s=0.5)
        fig.show()
        
        

    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()
    #input()

    exit()


if __name__ == '__main__':
    base = "C:\\Users\\saba saleemi\\Desktop\\UROP\\TESS\\transient_lcs\\unzipped_ccd\\" # Forced value of base
    set_base(base)


    # plot this only for main cluster
    sector = 43
    TOI = TOI_list(sector)
    plot_feat(sector)

    for toi in TOI:
        LC(sector,*toi).remove_outliers().plot()
        print(func(toi))




#from cluster_secondary import w_metric
#import lcobj

#tags = lcobj.TagFinder(tags)


##(1, 4, 1761, 1168) - scatters

##(3, 3, 206, 1442)
##(3, 2, 900, 594) - negative scatter

## compare balls to different sectors!!!

##import cleanup_anomaly
##vec1 = cleanup_anomaly.positive_scatter_candid
##vec1 = cleanup_anomaly.negative_scatter_candid
#vec1 = data[tags.find((3, 2, 1301,1298))]
#print(repr(vec1))
##print(w_metric(vec1,vec2))
##input('enter')
#f = []
#for vec2 in data:
   # f.append(w_metric(vec1,vec2))
#feature = np.array(f)#1/(1-data[:,6]**2)
##for i in range(len(data)):
##    point = tag[i]
##    if tuple(point.astype('int32')) in TOI:
##        print(point, feature[i])


#plt.ion()
#fig,ax = plt.subplots()

#import lcobj

#from scipy.stats import linregress

#try:
   # num_feat = feature.shape[1]
   # for i in range(num_feat):
       # ax.scatter(range(feature.shape[0]),feature[:,i], s= 1,picker=5)
#except IndexError:
   # num_feat = 1
   # ax.scatter(range(feature.shape[0]),feature[:], s= 1,picker=5, c= TOI_color(feature))
   # #ax.scatter(range(feature.shape[0]),[0.47]*feature.shape[0],s=0.1)
   # #ax.scatter(range(feature.shape[0]),[1.0]*feature.shape[0],s=0.1)

#def onpick(event):
   # ind = event.ind
   # ccd_point = tags[ind][0]
   # coords = (int(ccd_point[0]),int(ccd_point[1]),int(ccd_point[2]), int(ccd_point[3]))
   # fig1,ax1 = plt.subplots()
   # lc = lcobj.LC(sector,*coords)
   # ax1.scatter(lc.time,lc.normed_flux,s=0.5)
   # lc.remove_outliers()
   # granularity = 1.0           # In days
   # bins = granularity*np.arange(27)
   # bin_map = np.digitize(lc.time-lc.time[0], bins)

   # feat = []
   # for bin in bins:
       # dp_in_bin = np.ma.nonzero(bin_map == bin+1)     
       # flux, time = lc.normed_flux[dp_in_bin], lc.time[dp_in_bin]
       # if flux.size in {0,1}:
           # slope ,c, r = 0,0,1
       # else:
           # slope, c, r = linregress(time,flux)[:3]  #(slope,c,r)
    
       # feat.append([slope, c,r**2])
   # fit = []
   # time = [ [t for t in lc.time if lc.time[0] + day <= t < lc.time[0]+day+1] for day in range(27)]
   # for day in range(27):
       # m,c,_ = feat[day]
       # fit += [m*t+c for t in time[day]]
   # print(coords)
   # slope, c = linregress(lc.time,lc.flux)[:2]
   # #print(data_raw[ind][0][-1])

   # #lc.pad_flux()
   # #lc.make_FFT()
   # ax1.scatter(lc.time,lc.normed_flux,s=0.5)
   # #ax1.scatter(lc.phase_space,lc.flux,s=0.5)
   # ax1.scatter(lc.time, fit,s=0.5)
   # #ax1.scatter(lc.time, slope*lc.time+c,s=0.5)


#fig.canvas.mpl_connect('pick_event', onpick)
#plt.show(block = False)
#input()





##from sklearn.preprocessing import StandardScaler
##scaler = StandardScaler()

##plt.scatter(range(len(feature)),scaler.fit_transform(feature.reshape(-1,1)), s= 1)
##plt.show()