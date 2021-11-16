import numpy as np
import warnings
from matplotlib import pyplot as plt
from sklearn import neighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn_extra.cluster import KMedoids
from pyclustering.cluster.kmedoids import kmedoids


# Algorithmes

# k-plus proches voisins
import adult


def knn(voisins, train_matrice, test_matrice, y_train, y_test):
    clf = neighbors.KNeighborsClassifier(n_neighbors=voisins, metric='precomputed', algorithm='brute')

    clf.fit(train_matrice, y_train)
    knn_train = clf.predict(train_matrice)
    knn_test = clf.predict(test_matrice)

    print(knn_train)
    print(knn_test)

    print("Le score de knn est " + str(clf.score(test_matrice, y_test)))


# k-médoides
def kmd(init, train_matrice, test_matrice):
    warnings.filterwarnings("ignore")
    initial_medoids = init
    kmedoids_instance = KMedoids(n_clusters=4, metric='precomputed', random_state=0)
    kmedoids_instance.fit(train_matrice, adult.y_train)

    closest_clusters_train = kmedoids_instance.predict(train_matrice)
    closest_clusters_test = kmedoids_instance.predict(test_matrice)

    fig = plt.figure(figsize=(12, 6))
    train_ax = fig.add_subplot(121)
    test_ax = fig.add_subplot(122)

    for i in range(4):
        train_cluster = adult.X_train[np.where(closest_clusters_train == i)[0]]
        train_ax.scatter(train_cluster[:, 0], train_cluster[:, 1])

        test_cluster = adult.X_test[np.where(closest_clusters_test == i)[0]]
        train_ax.scatter(test_cluster[:, 0], test_cluster[:, 1])
    plt.show()


# Isomap
def isomap(components, voisins, train_matrice, test_matrice):
    isomap = Isomap(n_components=components, n_neighbors=voisins, metric='precomputed')
    isomap_trainset = isomap.fit_transform(train_matrice)
    isomap_testset = isomap.transform(test_matrice)

    plt.show()

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(211)
    ax.set_title('Cosine Isomap sur circle')
    t = np.arange(len(isomap_trainset))
    ax.scatter(isomap_trainset, np.zeros_like(isomap_trainset), c= t)

    ax = fig.add_subplot(212)
    ax.set_title('Cosine Isomap sur infinity')
    t = np.arange(len(isomap_testset))
    ax.scatter(isomap_testset, np.zeros_like(isomap_testset), c= t)
    plt.show()


# PCoA
def pcoa(components, train_matrice, test_matrice):
    pcoa = KernelPCA(n_components=components, kernel='precomputed')
    pcoa_trainset = pcoa.fit_transform(-.5 * train_matrice ** 2)
    pcoa_testset = pcoa.transform(-.5 * test_matrice ** 2)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(211)
    ax.set_title('Cosine PCoA sur circle')
    t = np.arange(len(pcoa_trainset))
    ax.scatter(pcoa_trainset, np.zeros_like(pcoa_trainset), c=t)

    ax = fig.add_subplot(212)
    ax.set_title('Cosine PCoA sur infinity')
    t = np.arange(len(pcoa_testset))
    ax.scatter(pcoa_testset, np.zeros_like(pcoa_testset), c=t)
    plt.show()


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

    fig = plt.figure(figsize=(12, 6))
    circle_ax = fig.add_subplot(121)
    infinity_ax = fig.add_subplot(122)
    circle_ax.set_title('Cosine regroupement hiérarchique sur circle')
    infinity_ax.set_title('Cosine regroupement hiérarchique sur infinity')

    for i in range(5):
        circle_cluster = adult.X_train[np.where(agglo_trainset == i)[0]]
        circle_ax.scatter(circle_cluster[:, 0], circle_cluster[:, 1])

        infinity_cluster = adult.X_test[np.where(agglo_testset == i)[0]]
        infinity_ax.scatter(infinity_cluster[:, 0], infinity_cluster[:, 1])
    plt.show()

