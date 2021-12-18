from matplotlib import pyplot as plt
from numpy.core.shape_base import block
from accuracy_model import AccuracyTest, Data
import accuracy_model
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
from run_classif import base
from pathlib import Path
import plotly.express as px
import time

#base = "C:\\Users\\saba saleemi\\Desktop\\UROP\\TESS\\transient_lcs\\unzipped_ccd\\" # Forced value of base



lcobj.set_base(base)


def plotter():
    choice = input("plot all, plot one, plot result file, plot transients candidates: ")

    sector, cam, ccd = 32,4, 4

    path = Path(str(lcobj.gen_path(sector,cam,ccd,0,0))[:-6])
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
    elif choice == '1':
        name = None
        while True:
            name = input('what is the file? (0 to exit): ')
            if name == '0':
                break
            sec, cam,ccd,col, row = name.split()
            try:
                lc = lcobj.LC(sec,cam,ccd,col,row)
            except OSError:
                continue
            print(f'is_cleaned: {lc.iscleaned}')
            lc.plot()

    elif choice == '2':
        sector = input('file name: ')
        file_name = sector
        file_path = Path(f'Results/{file_name}.csv')
        data_api = Data(int(sector),'s')
        #feat_tag,feat_data = next(get_sector_data(sector,'s',verbose=False))
        feat_tag,feat_data = data_api.stags,data_api.get_all(type='scalar')

        raw_data = np.genfromtxt(file_path,delimiter = ',')
        tags = raw_data[:,:4].astype('int32')
        rand_tags = np.copy(tags)
        print(tags.shape)
        shuffle(rand_tags)

        stack: list[LC] = []
        for i,tag in enumerate(rand_tags):
            if 9*(len(rand_tags)//9)<=i:
                print(tag)
                LC(sector,*tag).remove_outliers().plot()
            if i>0 and i%9 ==0:
                fig, axs = plt.subplots(3, 3)
                for j,lc in enumerate(stack):
                    axs[j%3,j//3].scatter(lc.time,lc.flux,s=0.5)
                    axs[j%3,j//3].set_title(f'{lc.cam} {lc.ccd} {lc.coords}')
                #for ax in axs.flat:
                #    ax.set(xlabel='x-label', ylabel='y-label')

                    # Hide x labels and tick labels for top plots and y ticks for right plots.
                for ax in axs.flat:
                    ax.label_outer()
                fig.show() 
                plt.show()
                stack = []
            stack.append(LC(sector,*tag).remove_outliers())

        return None
        for i,tag in enumerate(rand_tags):

            print((i,int(sector), *tag))
            cam,ccd,col,row = tag
            
            found,i = False,0
            while not found:
                try:
                    lc = lcobj.LC(int(sector),cam,ccd,col,row)
                    sec = sector
                    found = True
                except OSError:
                    i+=1
            tag = np.array([cam,ccd,col,row]).astype('int32')
            lc.remove_outliers()
                     
            peaks = signal.find_peaks(lc.smooth_flux,prominence=6,distance=50)[0]
            lc.plot(show_smooth=True,show_bg=False)
            plt.show()
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
                lc.plot()


def remove_outliers(self: LC):
    #outlier_detector = IsolationForest(n_estimators=200,max_samples=1000,random_state=np.random.RandomState(314),contamination=0.01)  # dynamic percentage?
    #forest = outlier_detector.fit_predict(list(zip(self.normed_flux,self.normed_time)))
    #inliers = np.ma.nonzero(forest==1)[0]
    
    
    EPSILON = 0.02
    time = self.normed_time
    #time.resize(math.ceil(self.N/WIDTH)*WIDTH)
    #time.reshape(math.ceil(self.N/WIDTH),WIDTH)

    flux = self.normed_flux

    groups : dict= {0:{(time[0],flux[0])}}

    #block = [p for p in zip(time[:WIDTH],flux[:WIDTH])]
    block = [(time[0],flux[0],0)]

    for t,f in zip(time[1:],flux[1:]):
        #if len(block)==WIDTH:
        #    block.pop(0)
        while block and abs(block[0][0]-t)>EPSILON:
            block.pop(0)
        seen_groups =set()
        seen_points = set()
        for t_,f_,g_ in block:
            if abs(f_-f)>EPSILON:
                continue
            seen_points.add((t_,f_))
            seen_groups.add(g_)

        if len(seen_groups)>1:  # combines groups 
            new = set()
            for group in seen_groups:
                new |= groups[group]
                del groups[group]
            new.add((t,f))
            groups[min(seen_groups)] = new
            block = [(i,j,min(seen_groups) if (i,j) in new else k) for i,j,k in block]
            tfgroup = min(seen_groups)
        elif len(seen_groups) == 1:     # adds a to an existing group
            tfgroup = min(seen_groups)
            groups[tfgroup].add((t,f))
        else:   # creates a new group
            tfgroup = max(groups.keys())+1
            groups[tfgroup] = set([(t,f)])

        block.append((t,f,tfgroup))


    #for group in groups:

    return groups

def label(sector):
    data = Data(sector,'scalar')
    step = []
    tags = data.stags
    feat = data.get_all(type='scalar')[:,25]
    mask = [feat>1]
    for tag in tags[mask]:
        lc = LC(32,*tag).remove_outliers()
        fig,ax = plt.subplots()
        ax.scatter(lc.time, lc.flux, s = 5, picker=5)

        def onpick(event):
            ind = event.ind
            ccd_point = tags[ind][0]
            coords = (int(ccd_point[0]),int(ccd_point[1]),int(ccd_point[2]), int(ccd_point[3]))
            print((sector ,*coords))
            step.append((sector,*coords))

        fig.canvas.mpl_connect('pick_event', onpick)
        mng = plt.get_current_fig_manager()
        mng.frame.Maximize(True)
        plt.show()
    step = np.array(step).astype('int32')
    np.savetxt(Path(f'{sector}_step.csv'),step, fmt='%1d',delimiter =',')
if __name__ == '__main__':
    plotter()
    exit()
    tag = (45, 3, 4, 2041, 1979)
    tag = (45, 2, 2, 600, 285)
    tag = (40, 4, 1, 1786, 1373)
    #tag = (44, 4, 2, 931, 1763)
    #tag = (45, 3, 3, 1215, 506)
    LC(*tag).plot().remove_outliers().plot()
    #exit()
    tags = [
    (39, 4, 2, 556, 1699),
    (39, 3, 4, 1754, 65),
    (39, 3, 4, 1307, 257),
    (39, 3, 4, 1936, 166),
    (39, 3, 1, 1731, 1686)]

    tags = [
        (43, 3, 4, 1836, 1634),
        (45, 3, 2, 128, 662),
        (45, 3, 3, 1193, 1746), #question, is this change in brightness interesting?
        (45, 3, 1, 1008, 1266),
    ]
    #sector = 43
    #data = Data(sector,'scalar')
    #tags = data.stags
    for tag in tags:
        print(tag)
        LC(*tag).remove_outliers().plot()
    exit()
    #tag = 42,4,2,1581, 1521
    #tag = 43, 3, 2, 610, 574
    #tag = 43, 3, 2, 602, 553
    #data = Data(32,'scalar')
    #model = AccuracyTest(data.stags[:99,:])
    #ind,tags = model.insert(99)
    #np.random.shuffle(tags)
    #for tag in tags:#,accuracy_model.transient_tags:
    #    lc = LC(32,*tag)
    #    print(tag)
    #    lc.remove_outliers().plot()
    #    lc.pad_flux()
    #    lc.make_FFT()
    #    lc.plot_FFT()
    #    lc.plot_phase()

    #plotter() #41 40
    #43, 3, 2, 610, 574
    #43, 3, 2, 602, 553
    tags = [(38, 4, 4, 1001, 990),
    (38, 4, 2, 1715, 943),
    (38, 4, 4, 1010, 998),
    (38, 4, 4, 846, 737),
    (38, 4, 4, 640, 937),
    (38, 4, 4, 900, 868),
    (38, 4, 4, 958, 614),
    (32, 1, 1, 310, 702)]
    #tags = [(43, 1, 1, 1016, 338)]
    #tags = Data(43,'s').stags
    #tags = np.concatenate((43*np.ones((len(tags),1)),tags),axis = 1).astype('int32')
    #print(tags)
    from accuracy_model import transient_tags
    for tag in transient_tags:
        lc = LC(32,*tag).plot(show_bg=False)
        groups = remove_outliers(lc)
        print(groups.keys())
        for g in groups.values():
            n = np.array(list(g))
            plt.scatter(n[:,0],n[:,1])
        plt.show()
        lc.remove_outliers().plot(show_bg=False)

    exit()
    stack: list[LC] = []
    for i,tag in enumerate(transient_tags):
        if 9*(len(transient_tags)//9)<i:
            print(tag)
            LC(-1,*tag).remove_outliers().plot()
        if i>0 and i%9 ==0:
            #plot shit 
            fig, axs = plt.subplots(3, 3)
            for j,lc in enumerate(stack):
                axs[j%3,j//3].scatter(lc.time,lc.flux,s=0.5)
                axs[j%3,j//3].set_title(f'{lc.cam} {lc.ccd} {lc.coords}')
            #for ax in axs.flat:
            #    ax.set(xlabel='x-label', ylabel='y-label')

                # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axs.flat:
                ax.label_outer()
            fig.show() 
            plt.show()
            stack = []
        stack.append(LC(-1,*tag).remove_outliers())
    
    #from binning import bin
    #for tag in transient_tags[1:2]:
    #    lc = LC(-1,*tag).remove_outliers()
    #    lc.plot(flux=lc.normed_flux,time=lc.normed_time,show_bg=False)
    #    bin((-1,*tag))
    #plotter()