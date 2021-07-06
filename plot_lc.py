import os
import lcobj
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
def get_coords_from_path(file_path):
    # returns coords from any file path
    i = file_path.rfind('lc_')
    d = file_path.rfind('.')

    x = file_path[i+3:d]
    y = file_path[d+1:]
    return (x,y)


if __name__ == '__main__':
    choice = input("plot all? y/n: ")

    # 21 1 3 980 1248

    # Good:

    #32 2 4 1860 975


    sector, cam, ccd = 32, 2, 1

    path = lcobj.gen_path(sector,cam,ccd,0,0)[:-6]
    if choice == 'y':
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
    else:
        name = None
        while True:
            name = input('what is the file name? (0 to exit): ')
            if name == '0':
                break
            cam,ccd,col, row = name.split()
            try:
                lc = lcobj.LC(sector,cam,ccd,col,row)
            except OSError:
                continue
            lc.smooth_plot()            
            lc.plot()