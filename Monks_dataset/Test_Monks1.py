#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#2.1. Veri Yukleme
veriler = pd.read_csv('monks-problems-1_test.csv')
veriler2= pd.read_csv('monks-problems-1_train.csv')

x = veriler.iloc[:,1:7].values 
y = veriler.iloc[:,0].values 
z= veriler2.iloc[:,1:7].values
w=veriler2.iloc[:,0].values


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
z_train = sc.fit_transform(z)
X_test = sc.transform(x)

# Buradan itibaren sınıflandırma algoritmaları başlar


# 1. KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
knn.fit(z_train,w)

y_pred = knn.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y,y_pred)
print('KNN')
print(cm)



# 2. Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(z_train, w)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y,y_pred)
print('GNB')
print(cm)

# 3. Decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(z_train,w)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y,y_pred)
print('DTC')
print(cm)

#4. Neural Network 
from sklearn.neural_network import MLPClassifier

mlp= MLPClassifier()

mlp.fit(z_train,w)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y,y_pred)
print('MLP')
print(cm)
