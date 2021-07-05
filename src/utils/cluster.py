import numpy as np
from sklearn.cluster import KMeans


# use k-means to get clusters
def cluster(states, num_clusters):
    clas = KMeans(n_clusters=num_clusters)
    clas.fit(states)
    centers = clas.cluster_centers_ 
    labels = clas.labels_   

    s_clus_list = []
    s_clus_size = []
    for clas_ind in range(num_clusters):
        s_ind = np.where(labels==clas_ind)
        s_clus_list.append(states[s_ind])
        s_clus_size.append(states[s_ind].shape[0])
    

    return s_clus_list, s_clus_size