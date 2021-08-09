from TOI_gen import TOI_list
import numpy as np
from lcobj import get_sector_data, set_base
import matplotlib.pyplot as plt
from scipy import stats
import lcobj
import concurrent.futures


def func(tag):
    lc = lcobj.LC(37,*tag).remove_outliers()
    delta = lc.flux -lc.smooth_flux
    return stats.anderson(delta)[0]
def plot_feat(sector):
    tags, data = next(get_sector_data(sector, 's',verbose=False))


    def TOI_color(data):
        ret = []
        for i in range(len(data)):
            point = tags[i]
            if tuple(point.astype('int32').tolist()) in TOI:
                ret.append('red')
            else:
                ret.append('green')
        return np.array(ret)

    '''
    feat = np.array([amp,better_amp,med,mean,std,slope,r,skew,max_slope,\
    beyond1std, delta_quartiles, flux_mid_20, *flux_mid_35*, flux_mid_50, \
    flux_mid_65, flux_mid_80, *cons*, slope_trend, var_ind(near 2 means normal dist), med_abs_dev, \
    H1, R21, R31, Rcs, l , med_buffer_ran, perr,band_width, StetK, p_ander, day_of_i])
    '''

    with concurrent.futures.ProcessPoolExecutor() as executer:
        results = executer.map(func,tags)
        data = []
        for i,feat in enumerate(results):
            if i%10 == 0:
                print(i)
            data.append(feat)

    feat = data
    plt.ion()

    fig,ax = plt.subplots()
    ax.scatter(range(len(feat)),feat,s=0.5,picker=5,c=TOI_color(feat))
    def onpick(event):
        ind = event.ind
        ccd_point = tags[ind][0]
        coords = (int(ccd_point[0]),int(ccd_point[1]),int(ccd_point[2]), int(ccd_point[3]))
        fig1,ax1 = plt.subplots()
        lc = lcobj.LC(sector,*coords)
        #lc.remove_outliers()
        ax1.scatter(lc.time,lc.flux,s=0.5)
        print(coords)
        ax1.scatter(lc.time,lc.smooth_flux,s=0.5)

        granularity = 1.0           # In days
        bins = granularity*np.arange(27)
        bin_map = np.digitize(lc.time-lc.time[0], bins)

        interesting_d = []
        total_d = 0
        for bin in bins:#range(1,np.max(bin_map)+1):
            dp_in_bin = np.ma.nonzero(bin_map == bin+1)
            flux, time = lc.smooth_flux[dp_in_bin], lc.time[dp_in_bin]
            _, ind = np.unique(time, return_index=True)
            flux, time = flux[ind], time[ind]

            if flux.size in {0,1}:
                pass

            if np.mean(flux) > lc.std + lc.mean:
                interesting_d.append(1000)
            else:
                interesting_d.append(0)

            total_d +=1
        time = [ [t for t in lc.time if lc.time[0] + day <= t < lc.time[0]+day+1] for day in range(27)]
        fit = []
        for day in range(27):
            i = interesting_d[day]
            fit += [i for t in time[day]]
        ax1.scatter(lc.time, fit,s=0.5)
        ax1.scatter(lc.time, lc.N*[lc.mean+lc.std],s=0.5)
        
        

    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show(block = False)
    input()

    exit()


if __name__ == '__main__':
    base = "C:\\Users\\saba saleemi\\Desktop\\UROP\\TESS\\transient_lcs\\unzipped_ccd\\" # Forced value of base
    set_base(base)

    sector = 37
    TOI = TOI_list(sector)
    plot_feat(sector)




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
