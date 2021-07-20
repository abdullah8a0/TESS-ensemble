import os
import lcobj
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from lcobj import get_coords_from_path
base = "C:\\Users\\saba saleemi\\Desktop\\UROP\\TESS\\transient_lcs\\unzipped_ccd\\" # Forced value of base



lcobj.set_base(base)


if __name__ == '__main__':
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
    sector, cam, ccd = 32, 2, 1

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
        with open(file_path) as file:
            
            for i,tag in enumerate(file):
                print(i)
                cam,ccd,col,row = tag.split()
                
                found,i = False,0
                while not found:
                    try:
                        lc = lcobj.LC(int(sectors[i]),cam,ccd,col,row)
                        sec = sectors[i]
                        found = True
                    except OSError:
                        i+=1
                print(sec, tag)
                tag = np.array([cam,ccd,col,row]).astype('int32')
                print(feat_data[np.ma.nonzero([ (x==tag).all() for x in feat_tag]),-4])
                print(feat_data[np.ma.nonzero([ (x==tag).all() for x in feat_tag]),9])
                #lc.smooth_plot()            
                lc.plot()

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