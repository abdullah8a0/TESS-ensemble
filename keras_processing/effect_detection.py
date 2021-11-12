
#base = "C:\\Users\\saba saleemi\\Desktop\\UROP\\TESS\\transient_lcs\\unzipped_ccd\\" # Forced value of base
#set_base(base)
import tensorflow as tf
from pathlib import Path

RTS = data = HTP = None

class TagFinder:
    def __init__(self,tags_original):

        tags = np.copy(tags_original)

        tags = tags.tolist()
        tags = sorted(enumerate(tags), key =lambda x:x[1])
        map = {i:tags[i][0] for i in range(len(tags))}
        tags = np.array([x[1] for x in tags])                 # reorder feat with tags
        self.tags = tags
        self.map = map
    
    def find(self,tag):
        return self.map[self.bin_search(np.array(tag))]



    def gt(self,x,y):
        return self.lt(y,x)

    def bin_search(self,tag) -> int:
        tags = self.tags
        upper = tags.shape[0]-1
        lower = 0
        while lower <= upper:
            mid = (upper + lower)//2
            if self.lt(tags[mid], tag):
                lower = mid + 1
            elif self.gt(tags[mid] , tag):
                upper = mid - 1
            else:
                return mid
        raise Exception(f"Tag is not in the data : {tuple(tag)}")

    def lt(self,x,y):
        if (x==y).all():
            return False
        idx = np.where((x!=y))[0][0]
        #print(x[idx])
        return x[idx]< y[idx]



def is_hot(tag,feat):


    for i,line in enumerate(data):
        if (tag==line[:-2]) and line[4] in HTP:
            return True
    return False            

    if feat[7]>6:
        return True
    if feat[-10]<20:
        return True
    if feat[9] < 0.025:
        return True
    if feat[-3] < 0.4:
        return True
    if feat[0]/feat[1] > 10:
        return True

    return False

def is_RTS(tag,feat):
    for i,line in enumerate(data):
        if (tag==line[:-2]) and line[4] in RTS:
            return True

    return False

    if feat[-4]> 1.16: 
        return True
    if feat[11]> 0.44:
        return True
    return False

import numpy as np
def find_effects(sector, RTS_, HTP_):
    #RTS, HTP, data
    RTS = [int(i) for i in RTS_] if RTS_ is not None else []
    HTP = [int(i) for i in HTP_] if HTP_ is not None else []
    data = np.genfromtxt(Path(f'Processed/{sector}.txt'),delimiter=',').astype('int32')[:,:-1]
    tags,clus = data[:,:4],data[:,4]
    finder = TagFinder(tags)
    with open(Path(f"Effects/{sector}_RTS.txt"),'w') as rts, open(Path(f"Effects/{sector}_HTP.txt"),'w') as hot:
        for tag in tags:
            pass
            #if  (k := data[map[bin_search(tags,tag)]][4]) in HTP:
            #    np.savetxt(hot,tag.reshape((1,4)),delimiter = ',',fmt='%1.i')
            #    #hot.write('\n')
            #if k in RTS:
            #    #rts.write(str(sector)+ str(tag)[1:-1]+'\n')
            #    np.savetxt(rts,tag.reshape((1,4)),delimiter = ',',fmt='%1.i')

if __name__ == '__main__':
    print('hi')
    TagFinder