from matplotlib import pyplot as plt
from cleanup_anomaly import isdirty
from TOI_gen import TOI_list
from cluster_anomaly import hdbscan_cluster
import os
from scipy import signal, stats

from numpy.random import shuffle
import lcobj
import numpy as np
from lcobj import get_coords_from_path
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
    sector, cam, ccd = 35,2, 1

    path = lcobj.gen_path(sector,cam,ccd,0,0)[:-6]
    if choice == '0':
        user_in = None 
        with os.scandir(path) as entries:
            for i,entry in enumerate(entries):
                if not entry.name.startswith('.') and entry.is_file():
                    file_name = str(entry.name)
                    col, row = get_coords_from_path(file_name)
                    print(col,row)
                    try:
                        lc = lcobj.LC(sector, cam, ccd, col, row)
                    except:
                        continue
                    lc.plot()
                    lc.pad_flux()
                    lc.make_FFT()
                    print(1/lc.significant_fequencies[0][0], lc.significant_fequencies[0][1])
                    lc.plot_phase()
                    if user_in != '0':
                        user_in = input("Press Enter to continue, type 0 to plot all: ")
                        if user_in == '1':
                            break
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
            hist = np.histogram(lc.flux-lc.smooth_flux)
            plt.hist(f:=(lc.flux-lc.smooth_flux),bins='auto')
            plt.show()
            print(stats.anderson(f))
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
    exit()
#Must detect:  [  1   2 934 861]
#Must detect:  [   1    4 2036 1776]
#Must detect:  [   2    3 1312 1886]
#Must detect:  [   3    1 1565 1157]
#Must detect:  [   4    1  101 1641]
    from scipy.signal import find_peaks
    sec = 37

    lc = lcobj.LC(37,1,2,934,861).remove_outliers().plot(show_smooth=True,show_bg=False)
    lc.smooth_flux = signal.savgol_filter(lc.flux, 301, 2)
    
    print(len(peaks:=find_peaks(lc.smooth_flux,prominence=6,distance=50)[0]))
    lc.plot(show_smooth=True,show_bg=False,scatter=[(lc.time[peaks],lc.smooth_flux[peaks])])
    exit()


    for tag in TOI_list(sec):
        print(tag)
        lcobj.LC(sec,*tag).plot().remove_outliers().plot()


    #from cluster_anomaly import tsne_plot,scale_simplify

    #raw_data = np.genfromtxt('Results\\32.txt',delimiter=',')
    #tags,data = raw_data[:,:4],raw_data[:,-30:]
    #transformed_data = scale_simplify(data,True,15)   
    #clusterer = hdbscan_cluster(transformed_data,True,None,5,1,'euclidean')
    #tsne_plot(tags,transformed_data,clusterer.labels_,normalized=False) (35, 1, 3, 123, 721)

    exit()

    lc = lcobj.LC(32,3,2,900,594)
    lc.remove_outliers()
    lc.plot(show_bg=False)

    lc = lcobj.LC(35,1,2,1796,1368).plot()
    lc.remove_outliers()
    lc.plot(show_bg=False)



