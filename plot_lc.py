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

"""
def clip(table, base, sigma):
    # removes entries from 'table' where the 'base'
    # has variation more than 'sigma' 
    base_clip = astat.sigma_clip(base,sigma=sigma)
    return np.array([table[i] for i in np.ma.nonzero(base_clip)[0]])



    #### DEADZONE CHECK

    coords = get_coords_from_path(file_path)    # column,rows
    assert 43 < int(coords[0]) < 2091                                                # check for off by one
    assert int(coords[1]) < 2057
    print(f'CCD pixel: {coords}')

    ####
"""

if __name__ == '__main__':
    choice = input("plot all? y/n: ")

    sector, cam, ccd = 6, 1, 1

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
                        lc = lcobj.lc_obj(sector, cam, ccd, col, row)
                    except:
                        continue
                    lc.plot()
                    #lc.smooth_plot()
                    #lc.plot(flux=lc.flat(smooth=True))
                    #peaks = find_peaks(lc.smooth_flux, prominence=5, distance=10, width=10)[0]
                    #plt.scatter(lc.time[:len(lc.smooth_flux)],lc.smooth_flux, s= 1)

                    #plt.scatter(lc.time[peaks], lc.smooth_flux[peaks])
                    #plt.show()
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
            col, row = get_coords_from_path(name)
            lc = lcobj.lc_obj(sector,cam,ccd,col,row)
            #lc.make_periodogram()
            #print(lc.computed_freq_power)
            #plt.scatter(*lc.periodogram, s= 0.5)
            #plt.show()
            '''
            ind = np.argpartition(lc.periodogram[1], -10)[-10:]
            top_10_freq = lc.periodogram[0][ind]
            generic = []
            for i in range(10):
                if 95.8< top_10_freq[i] < 96.2:
                    generic.append(i)
                if 47.8 < top_10_freq[i] < 48.2:
                    generic.append(i)
            result = np.delete(top_10_freq,generic)

            print("Top ten: ", result)

            lc.plot_phase()
            '''
            #p_coeff = np.sum(lc.smooth_flux) 
            #print(p_coeff)
            lc.smooth_plot()            
            lc.plot()