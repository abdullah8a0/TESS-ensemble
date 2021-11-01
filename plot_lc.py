from matplotlib import pyplot as plt
from numpy.core.shape_base import block
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
        sectors = input('file name: ').split()
        file_name = '_'.join(sectors)
        file_path = Path(f'Results/{file_name}.txt')
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

if __name__ == '__main__':
    plotter()