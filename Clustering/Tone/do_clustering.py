import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering as clust_func
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

from utils import stand_regres

def get_toneFeature(tones):
    tones_range = np.max(tones, axis=-1, keepdims=True)-np.min(tones, axis=-1, keepdims=True)
    tones_diff = np.diff(tones, axis=-1)

    tones_features=np.concatenate((tones, tones_diff, 0.8*tones_range), axis=-1)

    return tones_features

def vote_n_clusters(idxes):
    result = Counter(idxes)
    max_cnt=0
    max_cnt_nClust=-1
    for i in range(len(result.keys())):
        if(result[i]>=1 and result[i]>max_cnt):
            max_cnt=result[i]
            max_cnt_nClust=i

    if(max_cnt<1 or max_cnt_nClust<0):
        max_cnt_nClust=np.median(np.array(idxes))

    return max_cnt_nClust

def get_simiMat(x, labels, n_clust, label_orders=None):
    x=get_toneFeature(x)

    result=np.zeros([n_clust, n_clust])
    if(label_orders==None):
        label_orders=list(range(n_clust))
    for i in range(n_clust):
        a=x[np.where(labels==i)]
        for j in range(n_clust):
            b=x[np.where(labels==j)]
            '''
            if(label_orders.index(j)==1):
                plt.plot(b[0,0:11])
                plt.show()
            '''
            result[label_orders.index(i),label_orders.index(j)]=np.mean(cosine_similarity(a,b))

    return result


def clustering_train(tones_all):
    #print(tones_all.shape)
    #tones_all_diff=np.diff(tones_all, axis=-1)
    tones_feature=get_toneFeature(tones_all)
    '''
    for i in range(tones_all.shape[0]):
        plt.plot(tones_all[i,:])
    plt.show()
    '''
    #轮廓系数
    silhouette_score_list = []
    #CH得分
    ch_score_list=[]
    #DB得分
    db_score_list=[]
    for k in range(3, 14):
        model = clust_func(n_clusters=k)
        model.fit(tones_feature)
        labels_pred=model.fit_predict(tones_feature)
        silhouette_score_list.append(silhouette_score(tones_feature, labels_pred))
        ch_score_list.append(calinski_harabasz_score(tones_feature, labels_pred))
        db_score_list.append(davies_bouldin_score(tones_feature, labels_pred))
        del model

    X = range(3, 14)

    fig, ax=plt.subplots(1,3,figsize=(16, 5))
    ax[0].set_xlabel('聚类数', fontdict={'family' : 'SimSun','size': 18})
    ax[0].set_ylabel('Silhouette score', fontdict={'family' : 'Times New Roman','size': 18})
    ax[0].plot(X, silhouette_score_list, 'o-')

    ax[1].ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    ax[1].set_xlabel('聚类数', fontdict={'family' : 'SimSun','size': 18})
    ax[1].set_ylabel('Calinski-Harabasz score', fontdict={'family' : 'Times New Roman','size': 18})
    ax[1].plot(X, ch_score_list, 'o-')

    ax[2].set_xlabel('聚类数', fontdict={'family' : 'SimSun','size': 18})
    ax[2].set_ylabel('Davies-Bouldin score', fontdict={'family' : 'Times New Roman','size': 18})
    ax[2].plot(X, db_score_list, 'o-')
    plt.show()
    del X

    '''
    print(np.diff(np.array(silhouette_score_list)))
    print(np.diff(np.array(ch_score_list)))
    print(np.diff(np.array(db_score_list)))
    '''

    best_n_clust=int(vote_n_clusters(
        [np.argmax(np.array(silhouette_score_list)),
         np.argmax(np.array(ch_score_list)),
         np.argmin(np.array(db_score_list)),
         np.argmin(np.diff(np.array(silhouette_score_list))),
         np.argmin(np.diff(np.array(ch_score_list))),
         np.argmax(np.diff(np.array(db_score_list)))])+3)
    #print(best_n_clust)
    model = clust_func(n_clusters=best_n_clust)
    #model = clust_func(n_clusters=4)
    model.fit(tones_feature)
    result = model.fit_predict(tones_feature)

    abnormal_idxes=list()
    sh_scores=silhouette_samples(tones_feature, result)
    for i in range(best_n_clust):
        idxes=np.where(result==i)
        max_shScore=max(sh_scores[idxes])
        abnormal_idxes.extend(np.where(sh_scores[idxes]<max_shScore*0.5)[0].tolist())

    return result, model, best_n_clust, abnormal_idxes

def clustering_predict(model, tones):
    #print(tones_all.shape)
    tones_feature = get_toneFeature(tones)
    result = model.fit_predict(tones_feature)

    return result
