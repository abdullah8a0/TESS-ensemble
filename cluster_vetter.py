from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from lcobj import LC, get_sector_data
import numpy as np


def vet_clusters(sector, tags, feat_data, clus_ind):
    processed = np.genfromtxt(f'Processed\\{sector}.txt', delimiter=',').astype('int32')


    pro_tags, pro_clus, labels = processed[:,:4][clus_ind],processed[:,4][clus_ind],-processed[:,5][clus_ind]

    assert (pro_tags == tags[clus_ind]).all()
    #print(pro_tags,ta)
    tags = tags[clus_ind]

    labels = np.where(labels ==-1, labels,pro_clus)

    #print(labels)

    transformed_data = feat_data[clus_ind]


    plt.ion()
    fig,ax = plt.subplots()


    data_tsne = TSNE(n_components=2).fit_transform(transformed_data)        ############# transformed or normed


    ax.scatter(data_tsne[:,0], data_tsne[:, 1], s = 5, picker=5, c= labels)#color_TOI(data_tsne, clusterer))#color_cluster(transformed_data,clusterer))       # what to color? 
    def onpick(event):
        ind = event.ind
        ccd_point = tags[ind][0]
        coords = (int(ccd_point[0]),int(ccd_point[1]),int(ccd_point[2]), int(ccd_point[3]))
        fig1,ax1 = plt.subplots()

        lc = LC(sector,*coords)
        sec = sector

        print((sec ,*coords))
        #print(data[ind][0])
        print(labels[ind])
        ax1.scatter(lc.time,lc.flux,s=0.5)
        #ax1.scatter(lc.time, lc.smooth_flux,s=0.5)


    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show(block = False)
    input()

