import concurrent.futures
from pathlib import Path
from matplotlib import rcParams
from scipy import stats
from scipy.sparse import construct
from accuracy_model import AccuracyTest, Data, to_index_map
import lcobj
import numpy as np


# implement decision tree as forwarding

def sec_isdirty(feat_s):
    #if feat_s[to_index_map['H1']] > 100:# and feat_s[-6]>1000:
    #    return True
     
    return False#find score at the end

def discriminative_scalar_tree(sfeat):
    votes = 0
    skew        = sfeat[to_index_map["skew"]]
    max_slope   = sfeat[to_index_map["max_slope"]]    
    flux_50     = sfeat[to_index_map["flux_mid_50"]]
    cons        = sfeat[to_index_map["cons"]]
    var_ind     = sfeat[to_index_map["var_ind"]]
    h1          = sfeat[to_index_map["H1"]]
    r31         = sfeat[to_index_map["R31"]]
    rcs         = sfeat[to_index_map["Rcs"]]
    l           = sfeat[to_index_map["l"]]
    band_width  = sfeat[to_index_map["band_width"]]
    stetk       = sfeat[to_index_map["StetK"]]


    if skew< -1.2:
        votes -=1
    elif skew<0:
        votes -= 0.5
    elif skew<1:
        votes +=1
    elif votes<2.5:
        votes +=1.5
    else:
        votes -=0.5


    if max_slope>3.5e+6:
        votes -=1
    elif max_slope>1.8e+6:
        votes += 0.5
    elif max_slope>1.0e+6:
        votes +=1
    else:
        votes += 0

    
    if flux_50>0.42:
        votes +=1.5
    elif flux_50>0.37:
        votes += 1
    elif flux_50>0.3:
        votes += 0
    else:
        votes -= 0.5


    if cons>0.028:
        votes += 0.5
    else:
        votes += 0

    if var_ind<0.274:
        votes += 1
    elif var_ind< 0.5:
        votes += 0.5
    elif var_ind<1:
        votes += 0
    else:
        votes -= 0.5

    if r31<0.8:
        votes -= 0.5
    elif r31<0.9:
        votes += 0
    elif r31<0.958:
        votes +=1
    else:
        votes += 0.5

    if rcs>0.34:
        votes +=1
    elif rcs>0.29:
        votes += 0.5
    elif rcs>0.2:
        votes += 0
    else:
        votes -= 1

    if l>291.3:
        votes += 1.5
    elif l>200:
        votes += 0.5
    else:
        votes += 0

    if band_width>20:
        votes -= 1
    elif band_width>7:
        votes += 0
    elif band_width>5:
        votes += 0.5
    elif band_width>1:
        votes += 1
    else:
        votes += 0

    if stetk>0.82:
        votes += 1.5
    elif stetk>0.8:
        votes += 0.5
    elif stetk>0.75:
        votes += 0
    else:
        votes -= 1
    
    return votes

def is_chatter(tag):
    lc = lcobj.LC(*tag).remove_outliers()
    #if ret and tag[1]==-1:
    #    print('chatter: ',tag,i)
    flux = lc.normed_flux
    L = len(flux)//50+1
    slotted:np.ndarray = np.resize(flux,50*L)
    slotted = slotted.reshape(L,50)
    upper = np.amax(slotted,axis=1)
    lower = np.amin(slotted,axis=1)
    var = np.var(upper-lower)

    return var>0.01367

def isdirty(features,tag):
    return (sec_isdirty(features) or is_chatter(tag))

def func(input):
    f_s,t = input
    output = isdirty(f_s,t)
    return (output,t)


def cleanup(tags=None,datafinder: Data=None,verbose=False):
    print('--Starting Cleanup--')
    #gen = lcobj.get_sector_data(sectors,'s',verbose=False)
    #gen_v = lcobj.get_sector_data(sectors,'v',verbose=False)

    #tags_s,feat_s = next(gen)
    #tags_v,feat_v = next(gen_v)
    #feat_s_tag_finder = lcobj.TagFinder(tags_s)
    #feat_v_tag_finder = lcobj.TagFinder(tags_v)
    #ind_s = []
    #ind_v = []
    #for i,tag in enumerate(tags):
    #    try:
    #        ind_s.append(feat_s_tag_finder.find(tag)) #map[bin_search(tags_s,tag)]
    #        ind_v.append(feat_v_tag_finder.find(tag))
    #    except Exception:
    #        continue

    scalar_feat_anom = datafinder.get_some(tags,type='scalar')
    data_in = ((feat_s,tuple((datafinder.sector,*c))) for feat_s,c in zip(scalar_feat_anom,tags))
    cleaned_tags = []
    with concurrent.futures.ProcessPoolExecutor() as executer:
        results = executer.map(func,data_in)
        data = []
        for i,data_out in enumerate(results):
            result, tag = data_out
            if i%100 ==0:
                print(i)
            if not result:
                data.append(tag)
        cleaned_tags = np.array(data)
    
    cleaned_tags = np.array(cleaned_tags)

    print("-- Final Results After Cleanup --")
    print(len(datafinder.stags),'->',a := tags.shape[0], '->',b := len(cleaned_tags))
    if verbose:
        print("Data Reduction: ",round(100*(1-b/a),1))
    return cleaned_tags[:,1:]
    #np.savetxt(Path(f'Results/{sectors[0]}.txt'), cleaned_tags[:,1:], fmt='%1.5e', delimiter=',')

if __name__ == "__main__":
    data_api = Data(32,'scalar',insert=[range(99)])
    tags = data_api.stags[-150:,:]
    model = AccuracyTest(tags)
    ind,tags = model.insert(99)
    cleanup(tags=tags,datafinder=data_api)
    
    pass