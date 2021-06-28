import numpy as np
from lcobj import base

data = np.genfromtxt(base + 'py_code\\count_transients_s1-34.txt')
#print(data)

def TOI_list(sector) -> list:
    sec_data = data[data[:,0]==sector][:,-4:].astype('int32')

    return sec_data.tolist()

if __name__ == '__main__':
    TOI = TOI_list(6)
    print(TOI)
