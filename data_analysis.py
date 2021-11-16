import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import neighbors, preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/adult.csv")

# Not relevent, not enough data
plt.hist(df['hours-per-week'], bins=range(100))
plt.show()
plt.hist(df['gender'], bins=range(5), orientation="horizontal")
plt.show()

plt.hist(df['age'], bins=range(100))
plt.show()

plt.hist(df['workclass'], bins=range(20), orientation="horizontal")
plt.show()

plt.hist(df['educational-num'], bins=range(20))
plt.show()

plt.hist(df['marital-status'], bins=range(10), orientation="horizontal")
plt.show()

plt.hist(df['occupation'], bins=range(20), orientation="horizontal")
plt.show()

plt.hist(df['relationship'], bins=range(10), orientation="horizontal")
plt.show()

# Mostly white
plt.hist(df['race'], bins=range(10), orientation="horizontal")
plt.show()

plt.hist(df['gender'], bins=range(5), orientation="horizontal")
plt.show()

plt.hist(df['capital-gain'], bins=range(100))
plt.show()

# Not relevent, not enough data
plt.hist(df['capital-loss'], bins=range(100))
plt.show()



# Mostly US
plt.hist(df['native-country'], bins=range(20), orientation="horizontal")
plt.show()

plt.hist(df['income'], bins=range(10), orientation="horizontal")
plt.show()


# plt.hist(df[df['gender'] == 'Male']['age'])
# plt.hist(df[df['gender'] == 'Female']['age'])
# plt.show()