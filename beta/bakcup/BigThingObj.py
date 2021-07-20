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

        self.strongly_periodic = pow > 100*2
        self.weakly_periodic = pow > 60*2

        _, r, _ = lc.linreg

        self.strongly_linear = 
        self.weakly_linear = 

        self.global_fit = (*lc.linreg, freq, pow)   #pow>100 then strongly periodic

        self.tokens = []

    def update_token(self):

        self.tokens.append()

        



TOI6 = [
    (1182,1232),
    (1363,1908),
    (886,1540),
    (945,1892),
    (1055,1543),
    (1491,708)
]

TOI21 = [
    (4, 1, 467,870),
    (3, 4, 190,709)
]

TOI32 =[
    (1, 3, 1953, 1208),
(2, 1, 1787, 866) ,
(2, 2, 1425, 411) ,
(2, 2, 1416, 1162),
(2, 3, 1109, 1645),
(2, 3, 1192, 362),
(2, 3, 216, 2030),
(3, 1, 1778, 738),
(3, 1, 137, 399),
(3, 1, 1032, 739),
(3, 1, 1584, 1125),
(3, 2, 100, 1095),
(3, 2, 1178, 2001),
(3, 2, 1103, 606),
(3, 4, 195, 1152),
(4, 1, 1752, 1579),
(4, 2, 70, 1634),
(4, 3, 957, 1933),
(4, 3, 1505, 1708),
(4, 3, 509, 256),
(4, 3, 1877, 1600),
(4, 3, 1044, 1626),
(4, 3, 939, 1874)
]

TOI33 = [
(1, 4, 645, 1136),
(3, 1, 558, 1874),
(3, 2, 1673, 647),
(3, 3, 450, 273),
(3, 3, 123, 1116),
(4, 4, 979, 1740),
]