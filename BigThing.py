from BigThingObj import *
import numpy as np
from lcobj import base,LC,get_sector_data

sector = 6




tag, data = get_sector_data(sector, 'v')
head_point = 0
while head_point < data.shape[0]:
    coords = tag[head_point]
    tail_point = head_point             # Head inclusive, Tail Exclusive
    while tail_point == data.shape[0] or tag[tail_point] == coords:
        tail_point += 1
    
    # Head and Tails are set

    lc = LC(sector,*coords)
    feat_vec = data[head_point:tail_point]

    token = Token(lc, feat_vec)

    slope, r, c, freq, pow = token.global_fit  # Linearfit, FFT fit

    self.strongly_periodic = pow > 100
    self.weakly_periodic = pow > 60
    preiod = 1/ freq

    for 





     
