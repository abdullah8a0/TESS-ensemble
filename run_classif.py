from TOI_gen import TOI_list
import cluster_anomaly
import lcobj
import feature_extract_vector
import feature_extract_scaler
import cleanup_anomaly
import numpy as np
import effect_detection
import os.path
import cluster_vetter


### settings ### 'None' asks for values while running

sector = 21         # 35 has a lot of reduction during cleanup
training_sector = None   # None makes "training sector = sector"
base = "C:\\Users\\saba saleemi\\Desktop\\UROP\\TESS\\transient_lcs\\unzipped_ccd\\"
model_persistence = False
show_TOI = True
tsne_all_clusters = False
tsne_results = False    #should remove
tsne_individual_clusters = False
vet_results = True      # implement
verbose = True
# the score function is messing up, even with most being accepted score is low???
# see what tags are ya passing in, check if renamer worked? (plot_lc works so it should have also)

################

lcobj.set_base(base)

def run_pipeline(sector,training_sector,model_persistence,tsne_all_clusters,tsne_results,tsne_individual_clusters,verbose,vet_results):

    sector = int(input("sector: ")) if sector is None else sector
    plot_tsne = input("plot clusters (time consuming) (y/n): ") == 'y' if tsne_all_clusters is None else tsne_all_clusters


    sector2 = str(sector) if int(sector) > 9 else '0'+str(sector)
    for cam,ccd in np.ndindex((4,4)):
        cam +=1
        ccd +=1
        if not os.path.isfile(f"Features\\features{sector2}_{cam}_{ccd}_s.txt"):
            feature_extract_scaler.extract_scaler_features(sector)
            break

    for cam,ccd in np.ndindex((4,4)):
        cam +=1
        ccd +=1
        if not os.path.isfile(f"Features\\features{sector2}_{cam}_{ccd}_v.txt"):
            feature_extract_vector.extract_vector_features(sector)
            break

    cluster_anomaly.set_plot_flag(plot_tsne)
    cluster_anomaly.set_sector(sector)
    cluster_anomaly.cluster_and_plot(min_size=15,min_samp=2,dim=15, verbose=verbose, vet_clus = tsne_individual_clusters,model_persistence=model_persistence, training_sector=training_sector)
    RTS_clusters = None
    HTP_clusters = None 
    if tsne_individual_clusters:
        RTS_clusters = input("Which cluster labels correspond to RTS? ").split()
        HTP_clusters = input("Which cluster labels correspond to hot pixels? ").split()


    effect_detection.find_effects(sector, RTS_clusters,HTP_clusters)
    
    cleanup_anomaly.set_sector(sector)
    cleanup_anomaly.cleanup(verbose=verbose)


    tsne_results = input("Show TSNE of results (y/n): ") == 'y' if tsne_results is None else tsne_results

    raw_data = np.genfromtxt(f'Results\\{sector}.txt', delimiter= ',')
    result_tags, sub_feat = raw_data[:,:4].astype('int32'), raw_data[:,-30:]
    transformed_data = cluster_anomaly.scale_simplify(sub_feat,False,15)
    if tsne_results:
        #cluster_anomaly.tsne_plot(result_tags,transformed_data,np.array([5]*len(result_tags)))
        cluster_anomaly.cluster_and_plot(sub_tags=result_tags, verbose=True,sub_feat=sub_feat,show_TOI=show_TOI, dim=15,min_size=7,min_samp=4,write=False,vet_clus=False)

    if vet_results:

        # DEANOMALIZATION HERE
        from hdbscan.prediction import all_points_membership_vectors
        clusterer,labels = cluster_anomaly.hdbscan_cluster(transformed_data,verbose,None,5,2,'euclidean')  #10,2
        labels_all = np.argmax(all_points_membership_vectors(clusterer),axis=1)
        cluster_anomaly.umap_plot(result_tags,transformed_data,labels_all,normalized=False)     #clusterer.labels_
        num_clus =  np.max(labels_all)
        clusters = np.array([np.ma.nonzero(labels_all == i)[0] for i in range(-1,1+num_clus)])
        #for i in range(num_clus+2):
        #    print(f"-- Showing result cluster {i-1} --")
        #    cluster_vetter.vet_clusters(/ssector,result_tags,transformed_data,clusters[i],processed= np.concatenate((result_tags,[[i] for i in clusterer.labels_],[[0] for i in range(len(result_tags))]),axis=1))
        bad_results = [int(i)+1 for i in input("Which clusters would you want removed?: ").split()] 
        
        if bad_results:
            to_be_removed = np.concatenate(clusters[bad_results])
        else:
            to_be_removed = []

        file = open(f"Results\\{sector}.txt", "r")
        lines = file.readlines()
        file.close()
        for index in sorted(to_be_removed, reverse=True):
            del lines[index]
        new_file = open(f"Results\\{sector}.txt", "w+")
        for line in lines:
            new_file.write(line)
        print(len(lines))
        new_file.close()

        print(f'Score: {cluster_anomaly.score(None,sector)}/{len(TOI_list(sector))}')


if __name__ == '__main__':

    np.seterr(all='ignore')
    run_pipeline(sector,training_sector,model_persistence,tsne_all_clusters,tsne_results,tsne_individual_clusters,verbose,vet_results)
    exit()
    #0 14 25 3 2 10 32
    raw_data = np.genfromtxt(f'Results\\{sector}.txt', delimiter= ',')
    result_tags, sub_feat = raw_data[:,:4].astype('int32'), raw_data[:,-30:]
    print(sub_feat[0])
    transformed_data = cluster_anomaly.scale_simplify(sub_feat,True,15)

    clusterer = cluster_anomaly.hdbscan_cluster(transformed_data,verbose,None,7,1,'euclidean')
    cluster_anomaly.tsne_plot(result_tags,transformed_data,clusterer.labels_)