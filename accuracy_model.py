import numpy as np
from lcobj import TagFinder, TagNotFound
import lcobj
from pathlib import Path

transient_tags = []
generated_transients = {}
# Remove transients from Renamer insertions.

def get_sector_data(sectors,t,verbose=True):
    assert t in ('scalar','vector','signat')

    if not hasattr(sectors, '__iter__'):
        sectors = [sectors]       

    for sector in sectors:
        sector2 = str(sector) if int(sector) > 9 else '0'+str(sector)
        data_raw =np.genfromtxt(Path(f"Features/{sector2}_{t}.csv"), delimiter=',')
        tags, data = data_raw[::,1:5].astype(int), data_raw[::,5:]
        yield tags,data

class Data:
    def __init__(self,sector,default,insert = []) -> None:
        self.sector = sector
        vec = get_sector_data(sector,'vector',verbose=False)
        scal = get_sector_data(sector,'scalar',verbose=False)
        sign = get_sector_data(sector, 'signat', verbose=False)
        self.vtags, self.vdata = next(vec)
        self.stags,self.sdata = next(scal)
        self.signattags,self.signatdata = next(sign)
        self.tagfindscalar = TagFinder(self.stags)
        self.tagfindvector = TagFinder(self.vtags)
        self.tagfindsignat = TagFinder(self.signattags)
        self.transientfind = TagFinder(transient_tags)
        self.default_type = default
        self.ind = insert
        #########################################
        # Get transient data here
        #########################################
        self.scalartran = np.array([])
        self.vectortran = np.array([])
        self.signattran = np.array([])
    
    def update_inset(self,insert):
        self.ind = np.concatenate((self.ind,insert))
        return self
    def new_insert(self,insert):
        self.ind = np.array(insert)
        return self

    def get(self,tag,verbose = False, type = None):
        if type is None:
            type =self.default_type
        if tag[0]==5:
            if type == 'scalar':
                ind = self.transientfind.find(tag)
                if ind in self.ind:
                    return self.scalartran[ind]
                else:
                    raise Exception('Data Finder tried to access a tag the it was not allowed to')
            elif type == 'vector':
                ind = self.transientfind.find(tag)
                if ind in self.ind:
                    return self.vectortran[ind]
                else:
                    raise Exception('Data Finder tried to access a tag the it was not allowed to')
            elif type == 'signat':
                ind = self.transientfind.find(tag)
                if ind in self.ind:
                    return self.signattran[ind]
                else:
                    raise Exception('Data Finder tried to access a tag the it was not allowed to')
            else:
                raise Exception('Wrong type')

        try:
            if type == 'scalar':
                return self.sdata[self.tagfindscalar.find(tag)]
            elif type == 'vector':
                return self.vdata[self.tagfindvector.find(tag)]
            elif type == 'signat':
                return self.signatdata[self.tagfindsignat.find(tag)]
            else:
                raise Exception('Wrong type')
        except TagNotFound:
            print('Tag not in sector')

    def get_all(self, type = None):
        if type is None:
            type = self.default_type
        if type == 'scalar':
            sector_data = self.sdata
            tran_data = self.scalartran[self.ind]##### get transient data ###
            ret = np.concatenate((sector_data,tran_data)) if len(self.ind) != 0 else sector_data
            return ret
        elif type == 'vector':
            sector_data = self.vdata
            tran_data = self.vectortran[self.ind]##### get transient data ###
            ret = np.concatenate((sector_data,tran_data)) if len(self.ind) != 0 else sector_data
            return ret
        elif type == 'signat':
            sector_data = self.signatdata
            tran_data = self.signattran[self.ind]##### get transient data ###
            ret = np.concatenate((sector_data,tran_data)) if len(self.ind) != 0 else sector_data
            return ret
        else:
            raise Exception('Wrong type')
    def get_some(self,tags,type=None):
        if type is None:
            type = self.default_type
        try:
            return np.array([self.get(tag,type=type) for tag in tags])
        except TagNotFound:
            print('Tag not in sector')

    

class AccuracyTest:     # Generative vs Discriminative Model
    def __init__(self, pre_tags) -> None:
        self.tags = pre_tags
    def insert(self):
        pass
    def measure(self,target = None, tags= [], trials = 1, seed = None,*args, **kwargs):
        assert target is not None
        target()

        pass
    def clean(self,post_tags) -> np.ndarray:
        pass



if __name__ == '__main__':
    data = Data(32,'s')
    print(data.get((1,1,110,956)))
    pass