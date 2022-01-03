from cluster_anomaly import cluster_and_plot,scale_simplify
import numpy as np
import concurrent.futures
from accuracy_model import Data,AccuracyTest
import hdbscan


def hdbscan_cluster(transformed_data,training_sector, min_size,min_samp,metric,epsilon=0):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size,cluster_selection_epsilon=epsilon ,min_samples=min_samp,metric=metric,prediction_data=True)       # BEST is 15,12 cluster size, 19 previous, 7 prev, 8 WORKS
        clusterer.fit(transformed_data)
        return clusterer.labels_


def func(arg):
    size,samp,transformed_data = arg
    center = (size,samp)
    labels = hdbscan_cluster(transformed_data,None,size,samp,'euclidean')
    num_clus =  np.max(labels)  #excluding anomalies and zero indexed
    if num_clus < 1:
        print(f'\t{size}, {samp}\t: No Clustering')
        return None

    clus_count = [np.count_nonzero(labels == i) for i in range(-1,1+num_clus)]

    return (center,num_clus,clus_count)


def descend(tags:np.ndarray,data_api:Data=None)->tuple:
    centers = [ # 6x6
        (15,5),
        (15,11),
        (15,17),
        (10,11),
        (10,5),
    ]
    data = data_api.get_some(tags,type='scalar')

    transformed_data = scale_simplify(data,False,15)

    # Removal of main types of Light Curves
    print("<<Starting Main Structure Removal>>")
    descended = False
    region = 0
    global_stats = {}   # center -> MAX_BELOW, MIN_ABOVE
    REGION = (0.55,0.72)
    num_decents = 0
    while not descended and num_decents<3:
        print(f"Descent number: {num_decents+1}")
        print(f"\tSearching region: {region}")
        #print(f"{REGION=}")
        cen_size,cen_samp = centers[region]
        stats = {}

        params_vals = [(i,j,transformed_data) for i in range(cen_size-3,cen_size+3) for j in range(cen_samp-3,cen_samp+3) if i > 0 and j>0] 
        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executer:
            results = executer.map(func,params_vals)
            data = []
            for i,feat in enumerate(results):
                if feat is not None:
                    data.append(feat)

        for (size,samp),num_clus,clus_count in data:
            anom ,struct = clus_count[0],clus_count[1:]
            struct.sort(reverse=True)
            passthrough = struct[0]*0.1 + anom
            reduction = 1-(passthrough/(sum(struct)+anom))
            stats[(size,samp)] = (reduction,anom,struct)

        # (reduction, parameters)
        IN = []
        MAX_BELOW = (0,(-1,-1)) 
        MIN_ABOVE = (1,(-1,-1))
        for param,(reduc,_,_) in stats.items():
            if MAX_BELOW[0]<reduc<REGION[0]:
                MAX_BELOW = (reduc,param)
            elif REGION[0]<=reduc<=REGION[1]:
                IN.append((reduc,param))
            elif REGION[1]<reduc<MIN_ABOVE[0]:
                MIN_ABOVE = (reduc,param)

        # find a good param
        # if cant ,go to new center
        # if last center then check global_stats for good enough param
        
        if IN:  # There is good clustering in region
            MID = (REGION[0]+REGION[1])/2
            best = (REGION[0],(-1,-1))
            for reduc,param in IN:  # Find best param
                if abs(reduc - MID) <= abs(best[0] - MID):
                    best = (reduc,param)
            curr_param=(best[1],stats[best[1]])
        else:
            global_stats[(cen_size,cen_samp)] = (MAX_BELOW,MIN_ABOVE)
            if region < 4:  # GOTO next region
                region +=1
                continue
            else:           # at last region
                BELOWS, ABOVES = [], []
                for below,above in global_stats.values():
                    if below[0]!= 0:
                        BELOWS.append(below)
                    if above[0] != 1:
                        ABOVES.append(above)
                
                if not BELOWS:
                    if not ABOVES:
                        raise ValueError("There is no clustering in any point in the parameter space, this probably means that we do not have enough data in this sector to draw conclusions")
                    else:   # Find the best param from ABOVES
                        best = min(ABOVES,key=lambda x:x[0])
                        # recalc stats to put in curr_param

                        labels = hdbscan_cluster(transformed_data,None,best[1][0],best[1][1],'euclidean')
                        num_clus =  np.max(labels)  #excluding anomalies and zero indexed
                        clus_count = [np.count_nonzero(labels == i) for i in range(-1,1+num_clus)]
                        anom ,struct = clus_count[0],clus_count[1:]
                        struct.sort(reverse=True)
                        passthrough = struct[0]*0.1 + anom
                        reduction = 1-(passthrough/(sum(struct)+anom))
                        curr_param=(best[1],(reduction,anom,struct))

                else:       # Find the best param from BELOWS
                    best = max(BELOWS,key=lambda x:x[0])


                    labels = hdbscan_cluster(transformed_data,None,best[1][0],best[1][1],'euclidean')
                    num_clus =  np.max(labels)  #excluding anomalies and zero indexed
                    clus_count = [np.count_nonzero(labels == i) for i in range(-1,1+num_clus)]
                    anom ,struct = clus_count[0],clus_count[1:]
                    struct.sort(reverse=True)
                    passthrough = struct[0]*0.1 + anom
                    reduction = 1-(passthrough/(sum(struct)+anom))
                    curr_param=(best[1],(reduction,anom,struct))


        after_cluster = cluster_and_plot(tags=tags,size=best[1][0],samp=best[1][1],datafinder=data_api)
        
        
        num_decents+=1
        if len(after_cluster) <2000: # The good clustering results in sufficient removal
            descended = True
            print(f"Done with decent, found {curr_param}")
        else:                                               # there is need to remove further
            region = 0
            global_stats = {}
            tags = after_cluster
            data = data_api.get_some(tags=after_cluster)
            transformed_data = scale_simplify(data,False,15)
            print(f"Need to decend more, found: {curr_param}")
            if num_decents == 1:    # 0.55, 0.72
                REGION = (.20,.40)
            elif num_decents == 2:
                REGION = REGION
            elif num_decents == 3:
                REGION = REGION
            # TODO: update REGION
    
    # center blob removed, now only structure remains
    tags = after_cluster
    #return after_cluster


    #####################
    # STRUCTURE REMOVAL #
    #####################
    
    print("<<Starting Sub-Structure Removal>>")

    centers = [ # 6x6
        (10,5),
        (10,11),
        (15,5),
        (15,11),
        (15,17),
    ]

    data = data_api.get_some(tags,type='scalar')

    transformed_data = scale_simplify(data,False,15)

    # Removal of sub-structures in data

    descended = False
    region = 0
    global_stats = {}   # center -> MAX_BELOW, MIN_ABOVE
    REGION = (0.10,0.20)    # percentage explained
    num_decents = 0

    while not descended and num_decents<2:
        print(f"Descent number: {num_decents+1}")
        #print(f"{REGION=}")
        print(f"\tSearching region: {region}")
        cen_size,cen_samp = centers[region]
        stats = {}
        for size,samp in ((i,j) for i in range(cen_size-3,cen_size+3) for j in range(cen_samp-3,cen_samp+3) if i > 0 and j>0):
            labels = hdbscan_cluster(transformed_data,None,size,samp,'euclidean')
            num_clus =  np.max(labels)  #excluding anomalies and zero indexed
            if num_clus < 1:
                print(f'\t{size}, {samp}\t: No Clustering')
                continue
            else:
                #print(size,samp)
                pass

            clus_count = [np.count_nonzero(labels == i) for i in range(-1,1+num_clus)]

            anom ,struct = clus_count[0],clus_count[1:]
            struct.sort(reverse=True)
            passthrough = struct[0]*0.1 + anom
            reduction = 1-(passthrough/(sum(struct)+anom))
            stats[(size,samp)] = (reduction,anom,struct)
        

        # (explained,anom,struct, parameters)
        IN = []                                     ### IN is the list of `good params`
        MAX_BELOW = (0,0,[],(-1,-1))                     ### 
        MIN_ABOVE = (1,0,[],(-1,-1))                     ### The definition of `good parameters` has changed for structure
        for param,(_,anom,struct) in stats.items():     ### removal!!
            explained = sum(struct)/(anom+sum(struct))

            if MAX_BELOW[0]<explained<REGION[0]:        ### 
                MAX_BELOW = (explained,anom,struct,param)           ### 
            elif REGION[0]<=explained<=REGION[1]:       ### 
                IN.append((explained,anom,struct,param))            ### 
            elif REGION[1]<explained<MIN_ABOVE[0]:      ### 
                MIN_ABOVE = (explained,anom,struct,param)           ### 

        # find a good param
        # if cant ,go to new center
        # if last center then check global_stats
        
        if IN:  # There is good clustering in region
            best = (0,0,[],(-1,-1))
            for explained,anom,struct,param in IN:  # Find best param
                if len(struct)>len(best[2]):
                    best = (explained,anom,struct,param)
            curr_param=(best[-1],stats[best[-1]])

        else:
            global_stats[(cen_size,cen_samp)] = (MAX_BELOW,MIN_ABOVE)
            if region < len(centers)-1:  # GOTO next region
                region +=1
                continue
            else:           # at last region
                BELOWS, ABOVES = [], []
                for below,above in global_stats.values():
                    if below[0]!= 0:
                        BELOWS.append(below)
                    if above[0] != 1:
                        ABOVES.append(above)
                
                if not BELOWS:
                    if not ABOVES:
                        raise ValueError("There is no clustering in any point in the parameter space, this probably means that we do not have enough data in this sector to draw conclusions")
                    else:   # Find the best param from ABOVES
                        best = min(ABOVES,key=lambda x:len(x[2])) #     Best explainatory power
                        # recalc stats to put in curr_param

                        labels = hdbscan_cluster(transformed_data,None,best[-1][0],best[-1][1],'euclidean')
                        num_clus =  np.max(labels)  #excluding anomalies and zero indexed
                        clus_count = [np.count_nonzero(labels == i) for i in range(-1,1+num_clus)]
                        anom ,struct = clus_count[0],clus_count[1:]
                        struct.sort(reverse=True)
                        passthrough = struct[0]*0.1 + anom
                        reduction = 1-(passthrough/(sum(struct)+anom))
                        curr_param=(best[-1],(reduction,anom,struct))

                else:       # Find the best param from BELOWS
                    best = max(BELOWS,key=lambda x:len(x[2]))


                    labels = hdbscan_cluster(transformed_data,None,best[-1][0],best[-1][1],'euclidean')
                    num_clus =  np.max(labels)  #excluding anomalies and zero indexed
                    clus_count = [np.count_nonzero(labels == i) for i in range(-1,1+num_clus)]
                    anom ,struct = clus_count[0],clus_count[1:]
                    struct.sort(reverse=True)
                    passthrough = struct[0]*0.1 + anom
                    reduction = 1-(passthrough/(sum(struct)+anom))
                    curr_param=(best[-1],(reduction,anom,struct))


        #after_cluster = model.test(data_api_model=data_api,target=cluster_and_plot,num=99,trials =1, seed = 137 , **kwargs)
        after_cluster = cluster_and_plot(tags=tags,size=best[-1][0],samp=best[-1][1],datafinder=data_api)
        
        
        num_decents+=1
        if len(after_cluster) <900: # The good clustering results in sufficient removal
            descended = True
            print(f"Done with decent, found {curr_param}")
        else:                                               # there is need to remove further
            region = 0
            global_stats = {}
            tags = after_cluster
            data = data_api.get_some(tags=after_cluster)
            transformed_data = scale_simplify(data,False,15)
            print(f"Need to decend more, found: {curr_param}")
            if num_decents == 1:    # 0.10, 0.20
                REGION = (.00,.30)
            elif num_decents == 2:
                REGION = REGION
            elif num_decents == 3:
                REGION = REGION
    return after_cluster

        



if __name__ == '__main__':
    sector = 41
    data = Data(sector, 'scalar') 
    tags = data.stags  
    
    #descend(tags,data_api=data)
    model = AccuracyTest(tags)
    kwargs = {'data_api' : data}
    model.test(data_api_model=data,target=descend,num=99,trials =1, seed = 137 , **kwargs)