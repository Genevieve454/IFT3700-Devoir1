import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import neighbors
from sklearn.model_selection import train_test_split

df = pd.read_csv("adult.csv")


# Fonction de similarité

def fonction_similarite_adult(x, y):
    somme_similarites = 0
    for i in range(len(x)):
        if x[i] == -1 or y[i] == -1:
            continue
        somme_similarites += (10 - abs((x[i] / 10) - (y[i] / 10)))
    return somme_similarites


def fonction_dissimilarite_adult(x, y):
    # score_similarite = fonction_similarite_adult(x,y)
    # score_parfait = 10 * x(len)
    # return 1 - (score_similarite/score_parfait)
    print(x)
    somme = x[0, 0]*y[0, 0]
    return (x[0]*y[0]).sum(axis=2)

def cosine_similarity(x, y, *args, **kwargs):
    print((x*y).sum(*args, **kwargs))
    return (x*y).sum(*args, **kwargs)/np.sqrt((x*x).sum(*args, **kwargs) * (y*y).sum(*args, **kwargs))

def cosine_dissimilarity(x, y, *args, **kwargs):
    return 1 - cosine_similarity(x, y, *args, **kwargs)

def get_fonction_dissimilarite_adult_matrix(X, Y=None):
    Y = X if Y is None else Y
    return cosine_dissimilarity(X[:, None], Y[None, :], axis=2)


# pre-processing
# mapping_genre = {"Male": 0, "Female": 10}
# df['genre-num'] = df.applymap(lambda x : mapping_genre[x])
cts_columns = ['age', 'educational-num', 'hours-per-week']
X = df[cts_columns]
y = df['occupation']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.05, test_size=0.01, stratify=y,
                                                    random_state=0)

df = pd.DataFrame(X_train)
X_train = df.to_numpy()

# knn
def knn(voisins):
    clf = neighbors.KNeighborsClassifier(n_neighbors=voisins, metric=fonction_similarite_adult, algorithm='brute')
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.05, test_size=0.01, stratify=y,
                                                        random_state=0)
    clf.fit(X_train, y_train)
    print("Le score de knn est " + str(clf.score(X_test, y_test)))


# knn(5)
hello = get_fonction_dissimilarite_adult_matrix(X_train)
# print(hello)
# 10-abs(nbhx/10 - nbhy/10)
# 10 et 0 pour les colonnes binaires
# -1 si information n'existe pas'
