import matplotlib.pyplot as plt
from accuracy_model import Data,transient_tags
from lcobj import LC
from cluster_anomaly import scale_simplify,tsne_plot,umap_plot
import numpy as np
import concurrent.futures

WIDTH, HEIGHT = 30, 20

def bin(tag)->np.ndarray:
    im = np.zeros((HEIGHT,WIDTH),dtype='int32')
    lc = LC(*tag).remove_outliers()
    for x,y in zip(lc.normed_time,lc.normed_flux):
        dx, dy = int(x*WIDTH) if x!=1 else WIDTH-1,int(y*HEIGHT) if y!=1 else HEIGHT-1
        im[19-dy,dx] +=1
    #im = plt.imshow(im, cmap='hot', aspect=1) 
    #plt.show()
    return im

def classify(tags):
    data = []
    with concurrent.futures.ProcessPoolExecutor() as executer:
        results = executer.map(bin,tags)
        Data = []
        for i,feat in enumerate(results):
            print(i)
            if feat is not None and np.all(np.isfinite(feat)) and not np.any(np.isnan(feat)):
                Data.append(feat.flatten())
        Data = np.array(Data)
        data = Data if data == [] else np.concatenate((data,Data))
    print(data.shape)
    normed = scale_simplify(data,False,10,skip=True)
    umap_plot(tags[0,0],tags,normed,np.zeros(len(normed)),with_sec=True)

if __name__ == '__main__':
    sector = 42
    data = Data(sector,'s')
    tags = data.stags[:500,:]
    l = len(tags)
    tags_with_sector = np.concatenate((np.array([sector]*l).reshape(l,1),np.array(tags)),axis=1)
    classify(tags_with_sector)
    pass
