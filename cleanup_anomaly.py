from TOI_gen import TOI_list
import matplotlib.pyplot as plt
import lcobj
import numpy as np


def lt(x,y):
    if (x==y).all():
        return False
    idx = np.where((x!=y))[0][0]
    #print(x[idx])
    return x[idx]< y[idx]


def gt(x,y):
    return lt(y,x)

def bin_search(tags,tag) -> int:
    upper = tags.shape[0]
    lower = 0
    while lower <= upper:
        mid = (upper + lower)//2
        if lt(tags[mid], tag):
            lower = mid + 1
        elif gt(tags[mid] , tag):
            upper = mid - 1
        else:
            return mid
    raise Exception


def isdirty(feat,lc):
    #perr =1.4?
    if feat[-4] > 0.45: # 1.4 = 80, 0.45 = 136
        #print('perr')
        #print(feat[-4])
        return True
    # beyond1std <0.025
    if feat[9] < 0.025: # 0.025 = 101
        #print('beyond1std')
        #print(feat[9])
        return True
    
    if feat[0]/feat[1] > 10:
        return True
    
    return False


def set_sector(sec):
    global sectors
    sectors = [sec]

def cleanup(verbose=False):
    print('--Starting Cleanup--')
    TOI = TOI_list(sectors[0])
    #file_name = '_'.join(sectors)
    anomalies_tags = np.genfromtxt(f'Results\\{sectors[0]}.txt').astype(int)
    pointer = 0
    gen = lcobj.get_sector_data(sectors,'s',verbose=False)
    new_anom = []

    feat_tags,feat = next(gen)
    feat_tags = feat_tags.tolist()
    feat_tags = sorted(enumerate(feat_tags), key =lambda x:x[1])
    map = {i:feat_tags[i][0] for i in range(len(feat_tags))}
    feat_tags = np.array([x[1] for x in feat_tags])                 # reorder feat with tags
    detected = 0
    must_detect = 0
    for i,tag in enumerate(anomalies_tags):
    #    if i % 100 == 0:
    #        print(i)
        found,i = False,0
    #    while not found:
    #        try:
    #            lc = lcobj.LC(int(sectors[i]),*tag)
    #            sec = sectors[i]
    #            found = True
    #        except OSError as p:
    #            print(p)
    #            i+=1
    #    if i != pointer:
    #        pointer += 1
    #        print('switching over')
    #        feat_tags, feat = next(gen)
    #        feat_tags = feat_tags.tolist()
    #        feat_tags.sort()
    #        feat_tags = np.array(feat_tags)

        if tuple(tag) in TOI:
            detected +=1

        from cluster_anomaly import sector_32_must_detects

        if tuple(tag) in sector_32_must_detects:
            must_detect +=1


        lc = None
        ind = map[bin_search(feat_tags,tag)]
        features = feat[ind]
        if not isdirty(features,lc):
            new_anom.append(np.concatenate((tag.astype('float32'), features)))
            #lc =lcobj.LC(32,*tag)
            #lc.plot()
    if verbose:
        print("-- Final Results After Cleanup --")
    print(len(feat_tags),'->',a := anomalies_tags.shape[0], '->',b := len(new_anom))
    if verbose:
        print("Data Reduction: ",round(100*(1-b/a),1))
    print(f'Accuracy: {detected}/{len(TOI)}\t Compulsory Accuracy in Sector 32: {must_detect}/{len(sector_32_must_detects)}')

    np.savetxt(f'Results\\{sectors[0]}.txt', np.array(new_anom), fmt='%1.5e', delimiter=',')