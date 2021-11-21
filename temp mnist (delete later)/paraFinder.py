import numpy as np

from matplotlib import pyplot as plt
from sklearn import neighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.metrics import accuracy_score
from sklearn_extra.cluster import KMedoids  # pip install scikit-learn-extra

from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

# source du code : demo 3
def kmeanPredict(X):
    scores = []
    k_range = range(2,20)
    for k in k_range:
        y_pred = KMedoids(n_clusters=k).fit_predict(X)
        scores.append(silhouette_score(X, y_pred))

    plt.plot(k_range, scores)
    plt.xlabel('k')
    plt.ylabel('Score silhouette')
    plt.title('Score silhouette en fonction de k');
    plt.show()

# source du code : demo 2
def kNeighbors(X_train,y_train):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    score_train = []
    score_val = []
    for k in np.arange(2, 30):
        clf = neighbors.KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        score_train.append(clf.score(X_train, y_train))
        score_val.append(clf.score(X_val, y_val))

    plt.plot(np.arange(2, 30), score_train, color='red', label='train')
    plt.plot(np.arange(2, 30), score_val, color='blue', label='valid')
    plt.legend()
    #plt.plot(np.arange(10, 30), score_val[9:], color='blue')
    print(np.max(score_val))
    print(np.argmax(score_val) + 1)
