from lcobj import base,LC
import numpy as np

class Token:
    def __init__(self, lc:LC, feat_vec):
        self.lc = lc
        self.feat_vec = feat_vec

        sector, cam ,ccd = lc.sector, lc.cam,lc.ccd
        sector2 = str(sector) if sector > 9 else '0'+str(sector)
        data_scal = np.genfromtxt(base + f"py_code\\Features\\features{sector2}_{cam}_{ccd}_s.txt", delimiter=',')

        for i in range(len(data_scal)):
            if data_scal[i][:4] == np.array([cam,ccd,*lc.coords]):
                feat_scal = data_scal[i][4:]
                break
        
        self.feat_scal = feat_scal

        freq, pow = lc.significant_frequencies[0]

        self.strongly_periodic = pow > 100
        self.weakly_periodic = pow > 60

        _, r, _ = lc.linreg

        self.strongly_linear = 
        self.weakly_linear = 

        self.global_fit = (*lc.linreg, freq, pow)   #pow>100 then strongly periodic

        self.tokens = []

    def update_token(self):

        self.tokens.append()

        


