import numpy as np
import warnings

from sklearn import neighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn_extra.cluster import KMedoids  # pip install scikit-learn-extra


# Algorithmes

# k-plus proches voisins
def knn(voisins, train_matrice, test_matrice, y_train, y_test):
    clf = neighbors.KNeighborsClassifier(n_neighbors=voisins, metric='precomputed', algorithm='brute')

    clf.fit(train_matrice, y_train)
    knn_train = clf.predict(train_matrice)
    knn_test = clf.predict(test_matrice)

    print("Le rapport de précision pour knn est: \n" + classification_report(knn_test, y_test))


# k-médoides
def kmd(train_matrice, test_matrice, y_train):
    warnings.filterwarnings("ignore")
    kmedoids_instance = KMedoids(n_clusters=4, metric='precomputed', random_state=0)
    kmedoids_instance.fit(train_matrice, y_train)

    closest_clusters_train = kmedoids_instance.predict(train_matrice)
    closest_clusters_test = kmedoids_instance.predict(test_matrice)

    #TODO modifier ici-bas et ajouter algo qui verifie l'accuracy


# Isomap
def isomap(components, voisins, train_matrice, test_matrice):
    isomap = Isomap(n_components=components, n_neighbors=voisins, metric='precomputed')
    isomap_trainset = isomap.fit_transform(train_matrice)
    isomap_testset = isomap.transform(test_matrice)

    # TODO modifier ici-bas et ajouter algo qui verifie l'accuracy


# PCoA
def pcoa(components, train_matrice, test_matrice):
    pcoa = KernelPCA(n_components=components, kernel='precomputed')
    pcoa_trainset = pcoa.fit_transform(-.5 * train_matrice ** 2)
    pcoa_testset = pcoa.transform(-.5 * test_matrice ** 2)

    # TODO modifier ici-bas et ajouter algo qui verifie l'accuracy


# Regroupement hiérarchique
def agglomerative_clustering_predict(agglomerative_clustering, dissimilarity_matrix):
    average_dissimilarity = list()
    for i in range(agglomerative_clustering.n_clusters):
        ith_clusters_dissimilarity = dissimilarity_matrix[:, np.where(agglomerative_clustering.labels_ == i)[0]]
        average_dissimilarity.append(ith_clusters_dissimilarity.mean(axis=1))
    return np.argmin(np.stack(average_dissimilarity), axis=0)


def hierarchique(clusters, train_matrice, test_matrice):
    agglomerative_clustering = AgglomerativeClustering(n_clusters=clusters, affinity='precomputed', linkage='average')
    agglomerative_clustering.fit(train_matrice)
    agglo_trainset = agglomerative_clustering_predict(agglomerative_clustering, train_matrice)
    agglo_testset = agglomerative_clustering_predict(agglomerative_clustering, test_matrice)

    # TODO modifier ici-bas et ajouter algo qui verifie l'accuracy

