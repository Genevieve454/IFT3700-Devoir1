import algorithmes
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# Occupation classification. From physical (0) to intellectual (2)
occupation_classification = {
    "Adm-clerical": 2,
    "Armed-Forces": 0,
    "Craft-repair": 1,
    "Exec-managerial": 2,
    "Farming-fishing": 0,
    "Handlers-cleaners": 0,
    "Machine-op-inspct": 1,
    "Other-service": 1,
    "Priv-house-serv": 1,
    "Prof-specialty": 1,
    "Protective-serv": 0,
    "Sales": 2,
    "Tech-support": 2,
    "Transport-moving": 1
}

# Marital status classification
marital_status_classification = {
    "Divorced": 1,
    "Married-AF-spouse": 0,
    "Married-civ-spouse": 0,
    "Married-spouse-absent": 1,
    "Never-married": 1,
    "Separated": 1,
    "Widowed": 1
}

# Relationship classification
relationship_classification = {
    'Unmarried': 0,
    'Wife': 1,
    'Husband': 2,
    'Not-in-family': 3,
    'Own-child': 4,
    'Other-relative': 5
}


def prep(person):
    print(person)
    result = np.array([person[0] / 100,  # age, range from 0 to 100
                    ])  

    if person[1] != "?":
        result = np.append(result, occupation_classification[person[1]] / 2)  # occupation, range from 0 to 2
    else:
        result = np.append(result, 0.5)

    result = np.append(result, marital_status_classification[person[2]])
    result = np.append(result, relationship_classification[person[3]]/5)

    print(abs(result))
    return abs(result)


def fonction_similarite_adult(x, y):
    delta = 1 - abs(np.apply_along_axis(prep, 2, x) - np.apply_along_axis(prep, 2, y))
    return np.average(delta, 2)


# Fonction de dissimilarité
def fonction_dissimilarite_adult(x, y):
    return 1 - fonction_similarite_adult(x, y)


# Fonction pour obtenir une matrice de similarite
def get_fonction_dissimilarite_adult_matrix(X, Y=None):
    Y = X if Y is None else Y
    return fonction_dissimilarite_adult(X[:, None], Y[None, :])


# Jeu de données
df = pd.read_csv("data/adult.csv")
cts_columns = ['age', 'occupation', 'marital-status', 'relationship']
X = df[cts_columns]
y = df['gender']

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, test_size=0.2, stratify=y, random_state=0)
X_train = pd.DataFrame(X_train).to_numpy()
X_test = pd.DataFrame(X_test).to_numpy()

# Creation de la matrice de dissimilarité
train_matrice_dissimilarite = get_fonction_dissimilarite_adult_matrix(X_train)
test_matrice_dissimilarite = get_fonction_dissimilarite_adult_matrix(X_test, X_train)


def run_adult():
    algorithmes.kmd([0, 1, 2], train_matrice_dissimilarite, test_matrice_dissimilarite)
    algorithmes.isomap(1, 2, train_matrice_dissimilarite, test_matrice_dissimilarite)
    algorithmes.pcoa(1, train_matrice_dissimilarite, test_matrice_dissimilarite)
    algorithmes.hierarchique(5, train_matrice_dissimilarite, test_matrice_dissimilarite)
    algorithmes.knn(11, train_matrice_dissimilarite, test_matrice_dissimilarite, y_train, y_test)
