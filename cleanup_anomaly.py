from cluster_anomaly import score
import concurrent.futures
from TOI_gen import TOI_list
import matplotlib.pyplot as plt
from scipy import stats
import lcobj
import numpy as np




def sec_isdirty(feat_s):
    if feat_s[-7] > 0.45 and feat_s[-6]>100:
        print('1')
        return True
    if feat_s[9] < 0.025:
        print('2')
        return True
    if feat_s[0]<5000:
        print('3')
        return True 
    
    return False#find score at the end

def is_chatter(tag):
    lc = lcobj.LC(*tag).remove_outliers()
    delta = lc.flux -lc.smooth_flux
    ret = stats.anderson(delta)[0]>5
    if ret: 
        print('5')
    return ret





positive_scatter_candid = np.array([-4.74576e-01,  9.01978e-01, -1.30099e-01,  4.62044e-01,
       -2.77856e-02,  4.10880e-02,  3.50000e-02,  4.64921e-02,
       -1.82828e-02,  1.75761e-02, -8.89820e-03,  3.16017e-03,
       -1.45503e-02,  1.09555e-02,  6.41957e-03,  2.43061e-03,
       -3.08234e-02,  5.11660e-02, -7.27247e-03,  3.06970e-03,
       -7.46975e-03,  2.03511e-03,  0.00000e+00,  1.00000e+00,
       -7.69003e-01,  4.88771e-01, -2.63584e-01,  7.90140e-01,
       -5.36792e-02,  1.40660e-01, -3.10513e-02,  4.27798e-02,
       -1.93370e-02,  2.14037e-02, -8.33012e-03,  4.18591e-03,
       -4.63989e-03,  1.29990e-03,  4.13744e-03,  1.22528e-03,
        1.18102e-02,  7.61344e-03,  1.85228e-02,  2.23565e-02,
       -1.79399e-02,  1.88467e-02,  2.60256e-03,  2.14696e-04,
        5.38168e-03,  1.71624e-03, -1.16322e-03,  3.40949e-05,
        0.00000e+00,  1.00000e+00])
    
    
    
negative_scatter_candids = np.array([
    
        [ 2.17772e-01,  5.24074e-01,  1.79117e-01,  3.88415e-01,
        9.24874e-02,  2.16905e-01,  7.62003e-02,  1.74686e-01,
        2.03196e-02,  1.11981e-02,  4.87385e-02,  5.67993e-02,
        2.38197e-03,  1.58517e-04, -6.60478e-02,  1.11012e-01,
        6.08035e-02,  9.16065e-02, -1.66516e-02,  7.79809e-03,
        3.73246e-02,  4.15180e-02, -4.92340e-02,  4.28803e-03,
       -2.26684e-02,  4.61785e-04,  1.81653e-01,  5.03487e-01,
        8.30425e-03,  2.37130e-03,  5.37212e-03,  7.16668e-04,
        5.59331e-02,  8.50370e-02, -5.25737e-02,  6.66062e-02,
        1.22733e-02,  4.64643e-03, -1.86344e-02,  9.30609e-03,
        3.29440e-02,  2.89935e-02,  1.30944e-02,  5.53989e-03,
        6.70438e-03,  1.24987e-03, -4.68814e-02,  6.60875e-02,
       -3.82691e-02,  3.75395e-02, -8.48605e-02,  1.14401e-01,
        0.00000e+00,  1.00000e+00],
        
        [ 2.77974e-01,  5.22361e-01,  1.09766e-01,  1.49896e-01,
        5.02023e-02,  3.79815e-02,  1.47139e-02,  3.25924e-03,
       -4.28068e-02,  2.63928e-02, -3.08346e-02,  1.55942e-02,
       -4.41577e-02,  2.98812e-02,  3.11310e-02,  1.25982e-02,
       -5.40982e-03,  4.77746e-04, -1.55788e-02,  3.71710e-03,
        2.74877e-02,  1.37156e-02, -8.21671e-03,  2.45122e-04,
        1.08526e+00,  1.56669e-01,  2.10489e-01,  4.50222e-01,
        4.69331e-02,  4.07517e-02,  3.29039e-03,  2.37858e-04,
        3.47655e-02,  2.08692e-02, -1.21410e-02,  2.35380e-03,
        3.58622e-02,  2.18373e-02, -2.82407e-02,  1.51383e-02,
        1.83695e-02,  5.84933e-03, -1.41641e-02,  2.90376e-03,
        3.93441e-03,  2.49827e-04, -3.24183e-02,  2.02218e-02,
       -1.22151e-02,  2.49986e-03,  4.44661e-02,  1.31549e-02,
        0.00000e+00,  1.00000e+00]                                      #0.55
        
        ])


    
    
    
    
    

#m_p,r_p = np.arctan(positive_scatter_candid[::2]),positive_scatter_candid[1::2]
#m_n,r_n = np.arctan(negative_scatter_candid[::2]),negative_scatter_candid[1::2]

def w_metric(vec1,vec2):
    m1,r1 = vec1[::2],vec1[1::2]
    m2,r2 = vec2[::2],vec2[1::2]
    m1 = np.arctan(m1)
    m2 = np.arctan(m2)
    return sum(np.sqrt(np.abs(r2-r1)*(m1-m2)**2))


def vec_isdirty(feat_s):
    t =  w_metric(positive_scatter_candid,feat_s) <1 or w_metric(negative_scatter_candids[0],feat_s)<0.47 or w_metric(negative_scatter_candids[1],feat_s)<0.55
    if t: 
        print('4')
    return t

def isdirty(features,features_v,tag):
    return (sec_isdirty(features) or vec_isdirty(features_v) or is_chatter(tag))

def func(input):
    f_s,f_v,t = input
    output = isdirty(f_s,f_v,t)
    return (output,(t,f_v,f_s))


def set_sector(sec):
    global sectors
    sectors = [sec]

def cleanup(verbose=False):
    print('--Starting Cleanup--')
    TOI = TOI_list(sectors[0])
    anomalies_tags = np.genfromtxt(f'Results\\{sectors[0]}.txt').astype(int)
    gen = lcobj.get_sector_data(sectors,'s',verbose=False)
    gen_v = lcobj.get_sector_data(sectors,'v',verbose=False)
    new_anom = []

    tags_s,feat_s = next(gen)
    tags_v,feat_v = next(gen_v)
    feat_s_tag_finder = lcobj.TagFinder(tags_s)
    feat_v_tag_finder = lcobj.TagFinder(tags_v)
    ind_s = []
    ind_v = []
    for i,tag in enumerate(anomalies_tags):
        try:
            ind_s.append(feat_s_tag_finder.find(tag)) #map[bin_search(tags_s,tag)]
            ind_v.append(feat_v_tag_finder.find(tag))
        except Exception:
            continue
#           features = feat_s[ind_s]
#           features_v = feat_v[ind_v]
#           sec_tag = tuple((sectors[0],*tag))
#           if isdirty(features,features_v,sec_tag):
#               new_anom.append(np.concatenate((tag.astype('float32'), features_v,features)))
    #for j in ((feat_s[a],feat_v[b],tuple((sectors[0],c))) for a,b,c in zip(ind_s,ind_v,anomalies_tags)):
    
    data_in = ((feat_s[a],feat_v[b],tuple((sectors[0],*c))) for a,b,c in zip(ind_s,ind_v,anomalies_tags))
    with concurrent.futures.ProcessPoolExecutor() as executer:
        #results = executer.map(lambda p:isdirty(*p),)
        results = executer.map(func,data_in)
        data = []
        for i,data_out in enumerate(results):
            result, info = data_out
            if i%10 ==0:
                print(i)

            if not result:
                data.append(np.concatenate(info))
        new_anom = np.array(data)

    
    new_anom = np.array(new_anom)

    if verbose:
        print("-- Final Results After Cleanup --")
    print(len(tags_s),'->',a := anomalies_tags.shape[0], '->',b := len(new_anom))
    if verbose:
        print("Data Reduction: ",round(100*(1-b/a),1))
    s = score(new_anom[:,1:5].astype("int32"),sectors[0])
    print(f'Score: {s}/{len(TOI_list(sectors[0]))}')

    np.savetxt(f'Results\\{sectors[0]}.txt', new_anom[:,1:], fmt='%1.5e', delimiter=',')

if __name__ == "__main__":
    tags,feats = next(lcobj.get_sector_data(sec:=37,'s',verbose=False))
    tags_v,feat_v =next(lcobj.get_sector_data(sec, 'v',verbose=False))
    binsearch_tags = lcobj.TagFinder(tags)
    binsearch_tags_v = lcobj.TagFinder(tags_v)
    tag = (4,1,101,1641)
    for tag in TOI_list(sec):
        #lcobj.LC(37,*tag).smooth_plot()
        #lcobj.LC(37,*tag).remove_outliers().plot()
        ind = binsearch_tags.find(tag)
        ind_v = binsearch_tags_v.find(tag)
        print(tags[ind],func((feats[ind],feat_v[ind_v],(sec,*tag)))[0])