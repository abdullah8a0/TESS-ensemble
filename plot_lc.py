from matplotlib import pyplot as plt
from cleanup_anomaly import isdirty
from TOI_gen import TOI_list
from cluster_anomaly import hdbscan_cluster
import os
import math
from scipy import signal, stats
from sklearn.mixture import GaussianMixture
from numpy.random import shuffle
import lcobj
import numpy as np
from lcobj import LC, get_coords_from_path
base = "C:\\Users\\saba saleemi\\Desktop\\UROP\\TESS\\transient_lcs\\unzipped_ccd\\" # Forced value of base



lcobj.set_base(base)


def plotter():
    choice = input("plot all, plot one, plot result file, plot transients candidates: ")

    # 21 1 3 980 1248

    # Good:

    #32 2 4 1860 975
    
    #33 1 1 317 288
    #33 1 3 331 914 
    #33 1 1 370 794
    #33 1 2 489 1852
    #33 1 3 705 187
    #33 1 4 1745 535
    #33 1 4 1862 457
    #33 1 4 2076 833
    #33 1 4 2076 884
    #33 2 1 1783 1972
    #33 2 1 603 1408
    #33 2 3 1422 1754
    #33 2 4 443 376




    #34 [   1    2 2080 1395]
#34 [   1    2 2047 1124]
#34 [   1    2 2036 1312]
#34 [   1    2 2028 1026]
#34 [   1    2 2011 1311]
#34 [   1    2 2000 1003]
    sector, cam, ccd = 32,4, 4

    path = lcobj.gen_path(sector,cam,ccd,0,0)[:-6]
    if choice == '0':
        user_in = None 
        with os.scandir(path) as entries:
            for i,entry in enumerate(entries):
                if not entry.name.startswith('.') and entry.is_file():
                    file_name = str(entry.name)
                    col, row = get_coords_from_path(file_name)
                    print((sector,cam,ccd, col,row))
                    tag = (sector,cam,ccd,col,row)
                    lc = lcobj.LC(*tag).plot() 
                    #lc_opt = lcobj.LC(*tag).remove_outliers()#.plot(show_smooth=True,show_bg=False)
                    #lc.plot(show_bg=False,scatter=[(lc_opt.time,lc_opt.flux)])
                    #lc = lcobj.LC(sector, cam, ccd, col, row).remove_outliers()
                    
                    
                    #delta = lc.flux -lc.smooth_flux
                    #plt.hist(delta,bins=30)
                    #print(stats.anderson(delta))
                    #plt.show()

    elif choice == '1':
        name = None
        while True:
            name = input('what is the file? (0 to exit): ')
            if name == '0':
                break
            cam,ccd,col, row = name.split()
            try:
                lc = lcobj.LC(sector,cam,ccd,col,row)
            except OSError:
                continue
            #lc.smooth_plot()
            lc.plot()

    elif choice == '2':
        sectors = input('file name: ').split()
        file_name = '_'.join(sectors)
        file_path = f'Results\\{file_name}.txt'
        feat_tag,feat_data = next(lcobj.get_sector_data(sector,'s',verbose=False))

        raw_data = np.genfromtxt(file_path,delimiter = ',')
        tags = raw_data[:,:4].astype('int32')
        rand_tags = np.copy(tags)
        shuffle(rand_tags)
        for i,tag in enumerate(rand_tags):

            print((i,int(sectors[0]), *tag))
            cam,ccd,col,row = tag
            
            found,i = False,0
            while not found:
                try:
                    lc = lcobj.LC(int(sectors[i]),cam,ccd,col,row)
                    sec = sectors[i]
                    found = True
                except OSError:
                    i+=1
            tag = np.array([cam,ccd,col,row]).astype('int32')
            #print(feat_data[np.ma.nonzero([ (x==tag).all() for x in feat_tag]),-4])
            #print(feat_data[np.ma.nonzero([ (x==tag).all() for x in feat_tag]),9])
            lc.remove_outliers()
            #lc.smooth_plot()   
                     
            #lc.smooth_flux = signal.savgol_filter(lc.flux, 301, 2)
            peaks = signal.find_peaks(lc.smooth_flux,prominence=6,distance=50)[0]
            lc.plot(show_smooth=True,show_bg=False)
            #plt.hist(f:=(lc.flux-lc.smooth_flux),bins='auto')
            plt.show()
            #print(stats.anderson(f))
            #print(len(peaks))
                #(2, 2, 1184, 1908)
    elif choice == '3':
        sectors = input('file name: ')
        file_name = sectors + '_transients'
        file_path = f'{file_name}.txt'
        feat_tag,feat_data = next(lcobj.get_sector_data(sector,'s',verbose=False))
        with open(file_path) as file:
            
            for i,tag in enumerate(file):
                print(i)
                sec,cam,ccd,col,row = tag.split()
                lc = lcobj.LC(int(sec),cam,ccd,col,row)
                print(tag)
                tag = np.array([cam,ccd,col,row]).astype('int32')
                print(feat_data[np.ma.nonzero([ (x==tag).all() for x in feat_tag]),-4])
                print(feat_data[np.ma.nonzero([ (x==tag).all() for x in feat_tag]),9])
                #lc.smooth_plot()            
                lc.plot()
    elif choice == '4':

        file_name = input('file name: ')
        file_path = f'{file_name}.txt'
        feat_tag,feat_data = next(lcobj.get_sector_data(sector,'s',verbose=False))
        with open(file_path) as file:
            
            for i,tag in enumerate(file):
                print(i)
                try:
                    sec,cam,ccd,col,row = tag.split(',')
                except ValueError:

                    cam,ccd,col,row = tag.split(',')
                    sec = sector
                print(repr(row))
                lc = lcobj.LC(int(sec),cam,ccd,col,row[:-1])
                print(tag)
                tag = np.array([cam,ccd,col,row]).astype('int32')
                print(feat_data[np.ma.nonzero([ (x==tag).all() for x in feat_tag]),-4])
                print(feat_data[np.ma.nonzero([ (x==tag).all() for x in feat_tag]),11])
                #lc.smooth_plot()            
                lc.plot()


                #32 [   1    3 1860  806]
if __name__ == '__main__':
    plotter()
    #for tag in TOI_list(32):
    #    LC(32,*tag).remove_outliers().plot()
    #    print(tag)
    exit()
    lcobj.LC(*(32,3, 1, 1584, 1125)).plot()

# PResentation
    #LC(36, 4, 1, '1229', '1280').plot(show_smooth=True,show_bg=False) #chatter
# (36, 4, 1, '138', '461') ^
# (36, 4, 1, '1299', '1530') noisy

#normal case
    #LC(37, 1, 4, 1743, 1123).plot()
# (37, 1, 1, 254, 88)
# (37, 1, 4, 125, 116)
# (37, 1, 2, 709, 1955)
# (37, 2, 4, 160, 74)

# blue clus
#(37, 4, 4, 195, 214)
#[0]
#(37, 4, 4, 158, 107)
#[0 0 0 0]

#    exit()
    missed = [
        [   1,    1,  644, 1640],
        [   1,    3,  112, 1796],
        [   1,    4, 1125,  135],
        [   2,    1, 1156, 1336],
        [   3,    3, 1171,  881],
        [   3,    3,  133, 1392],
        [   3,    4,  706, 1062],
        [  4 ,  1, 377, 184],
        [   4,    1, 1922,  968],
    ]
    missed = TOI_list(37)
    for tag in missed:
        #tag = (37, 1, 4, 1667, 272)
        print(tag)
        lc = lcobj.LC(37,*tag).remove_outliers().plot(show_bg=False)
        continue

        fig1,ax1 = plt.subplots()
        ax1.scatter(lc.time,lc.flux,s=0.5)
        #ax1.scatter(lc.time,lc.smooth_flux,s=0.5)

        granularity = 1.0/3           # In days
        bins = granularity*np.arange(round(27/granularity))
        #print(bins)
        bin_map = np.digitize(lc.time-lc.time[0], bins)

        interesting_d = []
        total_d = 0
        for bin in bins:#range(1,np.max(bin_map)+1):
            dp_in_bin = np.ma.nonzero(bin_map == round(bin/granularity)+1)
            flux, time = lc.flux[dp_in_bin], lc.time[dp_in_bin]
            _, ind = np.unique(time, return_index=True)
            flux, time = flux[ind], time[ind]

            #print(bin,flux)
            #input()
            if flux.size == 0:
                interesting_d.append(0)
                continue
            if np.mean(flux) > lc.std + lc.mean:
                interesting_d.append(1000)
            else:
                interesting_d.append(0)

            total_d +=1
        time = [ [t for t in lc.time if lc.time[0] + hr8*granularity <= t < lc.time[0]+(hr8+1)*granularity] for hr8 in range(round(27/granularity))]
        fit = []
        #print(interesting_d)
        #print(time[1],len(time[1]))
        #input()
        for day in range(round(27/granularity)):
            i = interesting_d[day]
            fit += [i for t in time[day]]
        #ax1.scatter(lc.time, fit,s=0.5)
        #ax1.scatter(lc.time, lc.N*[lc.mean+lc.std],s=0.5)
        plt.show()



    exit()
    sec = 35
    tag = (35, 4, 1, 1922, 968)    
    tag = (37,   1,    2, 1012,  389)
    tag = (37,1,1,1101,88)
    tag = (36, 4, 1, '1229', '1280')

    tag=  (36, 4, 1, '138', '461')
    tag = (32,3, 4, 805, 1076)
    lc = lcobj.LC(*tag).plot(show_bg=False).remove_outliers()
    lc_opt = lcobj.LC(*tag).remove_outliers()#.plot(show_smooth=True,show_bg=False)
    lc.plot(show_bg=False)
    
    lc.plot(show_smooth=True,show_bg=False)
    lc.flatten().plot(flux=lc.flat)
    plt.hist(f:=(lc.flux-lc.smooth_flux),bins='auto')
    plt.show()
    print(stats.anderson(f))
    #print(len(peaks:=find_peaks(lc.smooth_flux,prominence=6,distance=50)[0]))
    #lc.plot(show_smooth=True,show_bg=False,scatter=[(lc.time[peaks],lc.smooth_flux[peaks])])
    exit()


    for tag in TOI_list(sec):
        print(tag)
        lcobj.LC(sec,*tag).plot().remove_outliers().plot()



