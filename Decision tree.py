import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks", color_codes=True)


import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'

data=pd.read_csv("zooclass.csv")
X = data.values[:, 1:17]
Y = data.values[:, 18]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(data.shape)

models = [
    ("LR", LogisticRegression()),
    ("LDA", LinearDiscriminantAnalysis()),
    ("KNN", KNeighborsClassifier()),
    ("DT", DecisionTreeClassifier()),
    ("NB", GaussianNB()),
    ("SVM", SVC())
]

# Modeller için 'cross validation' sonuçlarının  yazdırılması
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree

dt = tree.DecisionTreeClassifier(random_state=42)
dt.fit(X_train, Y_train)

tree.plot_tree(dt);

text_representation = tree.export_text(dt)
print(text_representation)

with open("decistion_tree.log", "w") as fout:
    fout.write(text_representation)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dt, 
                   class_names=['Amphibian','Bird','Bug','Fish','Invertebrate','Mammal','Reptile'], 
                   feature_names=['hair','feather','eggs','milk','airborne','aquatic','predator',
                                  'toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize'], 
                   filled=True)

fig.savefig("decistion_tree.png")
