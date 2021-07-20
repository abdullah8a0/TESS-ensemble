from BigThingObj import *
import numpy as np
from lcobj import base,LC,get_sector_data

sector = 6




tag, data = get_sector_data(sector, 'v')
head_point = 0
while head_point < data.shape[0]:
    coords = tag[head_point]
    tail_point = head_point             # Head inclusive, Tail Exclusive
    while tail_point < data.shape[0] and tag[tail_point] == coords:
        tail_point += 1
    
    # Head and Tails are set

    lc = LC(sector,*coords)
    feat_vec = data[head_point:tail_point]

    token = Token(lc, feat_vec)

    slope, r, c, freq, pow = token.global_fit  # Linearfit, FFT fit

    period = 1/ freq

    # add support for time sliced raw light curves'

    if token.strongly_linear:
        global_linear = True
        linearity_threshold = 0.00
    elif token.weakly_linear:
        global_linear = True
        linearity_threshold = 0.00
    else:
        global_linear = False
    
    for feat in feat_vec:
                        

        

    


    #end
    head_point=tail_point





     
