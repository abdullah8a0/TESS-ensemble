import cluster_anomaly
import lcobj
import cleanup_anomaly
import numpy as np
#import effect_detection
import os.path
from pathlib import Path
import accuracy_model
import feature_extract

### settings you need to care about ### 

sector = 38
#base = "C:\\Users\\saba saleemi\\Desktop\\UROP\\TESS\\transient_lcs\\unzipped_ccd\\"    #<- points to where the transient files are
base = Path('/Users/abdullah/Desktop/UROP/Tess/sector_data/transient_lcs')
#################





##########  'None' asks for values while running


training_sector = None 
model_persistence = False
show_TOI = False
plot_tsne = False
tsne_results = True
tsne_individual_clusters = False    # Set to true to find effects
vet_results = True   
verbose = False


################

lcobj.set_base(base)

def run_pipeline(sector,training_sector,model_persistence,tsne_all_clusters,tsne_results,tsne_individual_clusters,verbose,vet_results):

    #sector = int(input("sector: ")) if sector is None else sector
    #plot_tsne = input("plot clusters (time consuming) (y/n): ") == 'y' if tsne_all_clusters is None else tsne_all_clusters


    sector2 = str(sector) if int(sector) > 9 else '0'+str(sector)
    if not os.path.isfile(Path(f"Features/{sector2}_scalar.csv")):
        print('Generating scalar features')
        feature_extract.extract_scalar_features(sector)
    if not os.path.isfile(Path(f"Features/{sector2}_vector.csv")):
        print('Generating vector features')
        feature_extract.extract_vector_features(sector)
    if not os.path.isfile(Path(f"Features/{sector2}_signat.csv")):
        print('Generating signat features')
        feature_extract.extract_signat_features(sector)

    data_api = accuracy_model.Data(sector,default='scalar')

    # ENFORCE SUB TAGS/ REMOVE SECTOR
    #                                                      5n          5n+3
    
    before_tags = data_api.stags
    model = accuracy_model.AccuracyTest(before_tags)
    #ind,inserted_tags = model.insert(15)
    #data_api.new_insert(ind)
    
    #after_cluster = cluster_anomaly.cluster_and_plot(tags= inserted_tags,datafinder = data_api, verbose=verbose,plot_flag=plot_tsne, vet_clus = tsne_individual_clusters,model_persistence=model_persistence, training_sector=training_sector)
    
    #model.measure(data_api,after_cluster)
    #after_cluster = model.clean(after_cluster)
    #data_api.new_insert([])

    kwargs = {'datafinder' : data_api, 'verbose':verbose,'plot_flag':plot_tsne, 'vet_clus' : tsne_individual_clusters,\
        'model_persistence':model_persistence, 'training_sector':training_sector,'suppress':True}

    after_cluster = model.test(data_api_model=data_api,target=cluster_anomaly.cluster_and_plot,p=0.04,trials = 5, seed = 137 , **kwargs)

    exit()
    #RTS_clusters = None
    #HTP_clusters = None 
    #if tsne_individual_clusters:# para search after clean up
    #    RTS_clusters = input("Which cluster labels correspond to RTS? ").split()
    #    HTP_clusters = input("Which cluster labels correspond to hot pixels? ").split()
    
    
    #effect_detection.find_effects(sector)
    model = accuracy_model.AccuracyTest(after_cluster) 
    kwargs = {'datafinder':data_api,'verbose': verbose}
    cleaned_tags = model.test(data_api_model=data_api,target=cleanup_anomaly.cleanup,p=0.04,trials=1,seed=137,**kwargs)
    #cleaned_tags = cleanup_anomaly.cleanup(tags = after_cluster,datafinder=data_api,verbose=verbose)

    np.savetxt(Path(f'Results/{sector}.csv'),cleaned_tags, fmt='%1d',delimiter =',')
    
    result_tags = cleaned_tags
    sub_feat = data_api.get_some(tags= cleaned_tags, type='scalar')
    transformed_data = cluster_anomaly.scale_simplify(sub_feat,False,15)


    if vet_results:

        # DEANOMALIZATION HERE
        from hdbscan.prediction import all_points_membership_vectors

        size_base, samp_base = 5,3

        br = False
        while not br:
            for size,samp in [(i,j) for i in range(size_base-3,size_base+3) for j in range(samp_base-3,samp_base+3) if i > 0 and j>0]:
                clusterer,labels = cluster_anomaly.hdbscan_cluster(transformed_data,training_sector,size,samp,'euclidean')
                if max(labels) == -1:
                    continue
                labels_all = np.argmax(all_points_membership_vectors(clusterer),axis=1)
                print(f'{size}, {samp}\t: {(num:=np.max(labels_all))}')
                if 8<num<15:
                    br = True
                    break 
                
            if not br:
                lis =  [int(i) for i in input('None Found. Enter new center: ').split(' ')]
                size_base, samp_base = lis[0],lis[1]
                if len(lis) == 3:
                    size,samp = size_base,samp_base
                    br = True    
        print(f'{size}, {samp}')


        clusterer,labels = cluster_anomaly.hdbscan_cluster(transformed_data,None,size,samp,'euclidean')  #10,2
        labels_all = np.argmax(all_points_membership_vectors(clusterer),axis=1)
        cluster_anomaly.tsne_plot(data_api.sector,result_tags,transformed_data,labels_all,normalized=False)     #clusterer.labels_
        #num_clus =  np.max(labels_all)
        #clusters = np.array([np.ma.nonzero(labels_all == i)[0] for i in range(-1,1+num_clus)])
        #bad_results = [int(i)+1 for i in input("Which clusters would you want removed?: ").split()] 
        
        #if bad_results:
        #    to_be_removed = np.concatenate(clusters[bad_results])
        #else:
        #    to_be_removed = []

        #file = open(Path(f"Results/{sector}.txt"), "r")
        #lines = file.readlines()
        #file.close()
        #for index in sorted(to_be_removed, reverse=True):
        #    del lines[index]
        #new_file = open(Path(f"Results/{sector}.txt"), "w+")
        #for line in lines:
        #    new_file.write(line)
        #print(len(lines))
        #new_file.close()

if __name__ == '__main__':

    np.seterr(all='ignore')
    import time
    start = time.time()
    run_pipeline(sector,training_sector,model_persistence,plot_tsne,tsne_results,tsne_individual_clusters,verbose,vet_results)
    end = time.time()
    print(end-start)