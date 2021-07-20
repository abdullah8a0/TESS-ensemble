import cluster_anomaly
import lcobj
import feature_extract_scaler
import cleanup_anomaly
import numpy as np
import effect_detection
import os.path


### settings ### 'None' asks for values while running

sector = 32
training_sector = 32    # None makes "training sector = sector"
base = "C:\\Users\\saba saleemi\\Desktop\\UROP\\TESS\\transient_lcs\\unzipped_ccd\\"
model_persistence = False
tsne_all_clusters = True
show_results = False
tsne_results = True
tsne_individual_clusters = False
verbose = True

################

lcobj.set_base(base)

def run_pipeline(sector,training_sector,model_persistence,tsne_all_clusters,show_results,tsne_results,tsne_individual_clusters,verbose):

    sector = int(input("sector: ")) if sector is None else sector
    plot_tsne = input("plot clusters (time consuming) (y/n): ") == 'y' if tsne_all_clusters is None else tsne_all_clusters


    sector2 = str(sector) if int(sector) > 9 else '0'+str(sector)
    for cam,ccd in np.ndindex((4,4)):
        cam +=1
        ccd +=1
        if not os.path.isfile(f"Features\\features{sector2}_{cam}_{ccd}_s.txt"):
            feature_extract_scaler.extract_scaler_features(sector)
            break


    cluster_anomaly.set_plot_flag(plot_tsne)
    cluster_anomaly.set_sector(sector)
    cluster_anomaly.cluster_and_plot(min_size=8, verbose=verbose, vet_clus = tsne_individual_clusters,model_persistence=model_persistence, training_sector=training_sector)
    RTS_clusters = None
    HTP_clusters = None 
    if tsne_individual_clusters:
        RTS_clusters = input("Which cluster labels correspond to RTS?").split()
        HTP_clusters = input("Which cluster labels correspond to hot pixels?").split()


    effect_detection.find_effects(sector, RTS_clusters,HTP_clusters)
    
    cleanup_anomaly.set_sector(sector)
    cleanup_anomaly.cleanup(verbose=verbose)

    show_results = input("plot the results (y/n): ") == 'y' if show_results is None else show_results

    if show_results:
        file_path = f'Results\\{sector}.txt'
        feat_tag,feat_data = next(lcobj.get_sector_data(sector,'s',verbose=False))
        with open(file_path) as file:
            for i,tag in enumerate(file):
                print(i)
                cam,ccd,col,row = np.array([i for i in tag.split(',')[:4]],dtype='float').astype('int32')
                lc = lcobj.LC(sector,cam,ccd,col,row)
                print(sector, cam,ccd,col,row)
                tag = np.array([cam,ccd,col,row]).astype('int32')
                #print(feat_data[np.ma.nonzero([ (x==tag).all() for x in feat_tag]),-4])
                #print(feat_data[np.ma.nonzero([ (x==tag).all() for x in feat_tag]),9])
                #lc.smooth_plot()            
                lc.plot()

    tsne_results = input("Show TSNE of results (y/n): ") == 'y' if tsne_results is None else tsne_results

    if tsne_results:
        raw_data = np.genfromtxt(f'Results\\{sector}.txt', delimiter= ',')
        sub_tags, sub_feat = raw_data[:,:4].astype('int32'), raw_data[:,4:]
        cluster_anomaly.cluster_and_plot(sub_tags=sub_tags, sub_feat=sub_feat, dim=15,min_size=7,min_samp=1,write=False,vet_clus=False)


if __name__ == '__main__':
    run_pipeline(sector,training_sector,model_persistence,tsne_all_clusters,show_results,tsne_results,tsne_individual_clusters,verbose)

