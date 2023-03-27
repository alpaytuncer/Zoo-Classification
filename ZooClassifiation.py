import matplotlib.pyplot as plt
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


data=pd.read_csv("zooclass.csv")

print(data.shape)
data.head()

print(data.groupby("class").size())

mammal = data[data['class']=='Mammal']
amphibian = data[data['class']=='Amphibian']
bird = data[data['class']=='Bird']
bug = data[data['class']=='Bug']
fish = data[data['class']=='Fish']
invertebrate = data[data['class']=='Invertebrate']
reptile = data[data['class']=='Reptile']

#Visualization

sns.catplot(x="class", y="hair", hue="class", kind="bar", data=data)
sns.catplot(x="class", y="feathers", hue="class", kind="bar", data=data)
sns.catplot(x="class", y="eggs", hue="class", kind="bar", data=data)
sns.catplot(x="class", y="milk", hue="class", kind="bar", data=data)
sns.catplot(x="class", y="airborne", hue="class", kind="bar", data=data)
sns.catplot(x="class", y="aquatic", hue="class", kind="bar", data=data)
sns.catplot(x="class", y="predator", hue="class", kind="bar", data=data)
sns.catplot(x="class", y="toothed", hue="class", kind="bar", data=data)
sns.catplot(x="class", y="backbone", hue="class", kind="bar", data=data)
sns.catplot(x="class", y="breathes", hue="class", kind="bar", data=data)
sns.catplot(x="class", y="venomous", hue="class", kind="bar", data=data)
sns.catplot(x="class", y="fins", hue="class", kind="bar", data=data)
sns.catplot(x="class", y="legs", hue="class", kind="bar", data=data)
sns.catplot(x="class", y="tail", hue="class", kind="bar", data=data)
sns.catplot(x="class", y="domestic", hue="class", kind="bar", data=data)
sns.catplot(x="class", y="catsize", hue="class", kind="bar", data=data)

#Correlation Heatmap

plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Correlation Heatmap")
corr = data.corr()
sns.heatmap(corr, annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

#
g = sns.FacetGrid(data, col="class")
g.map(plt.hist, "legs")
plt.show()

#
data.plot(x="eggs",y="milk")

# ML
X = data.values[:, 1:17]
Y = data.values[:, 18]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.20)

print(X_train.shape)
print(X_test.shape)
print(data.shape)

print(data.groupby("class").size())


# Uygun modelin tespiti
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

from sklearn.tree import DecisionTreeClassifier
data = DecisionTreeClassifier()
data.fit(X_train, Y_train)
print("Score:", data.score(X_test,Y_test))

#İnsan için deneme yapıyoruz
X_yeni = [[1,0,0,1,0,0,1,1,1,1,0,0,2,0,1,1]]
tahmin = data.predict(X_yeni)
print(tahmin)


from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


clf_tree = DecisionTreeClassifier(criterion='gini')
clf_tree.fit(X_train, Y_train)

fig, ax = plt.subplots(figsize=(10, 10))
tree.plot_tree(clf_tree, fontsize=10)
plt.show()

from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
 
dot_data = export_graphviz(clf_tree, filled=True, rounded=True,
                                    class_names=['Mammal','Bird','Reptile','Fish','Amphibian','Bug','Invertebrate'],
                                    feature_names=['hair','feather','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize'],
                                    out_file=None)
graph = graph_from_dot_data(dot_data)

nb = GaussianNB()
nb.fit(X_train, Y_train)
nb.score(X_test,Y_test)

clf = LogisticRegression()
clf.fit(X_train, Y_train) 
clf.score(X_test, Y_test)

kn = KNeighborsClassifier()
kn.fit(X_train, Y_train)
kn.score(X_test, Y_test)