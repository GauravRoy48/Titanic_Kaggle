#####################################################################################
# Creator     : Gaurav Roy
# Date        : 11 June 2019
# Description : The code contains the various classification models to be applied on
#               the titanic dataset.
#####################################################################################

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset

# Dropping Cabin and Ticket columns due to irrelevant and insufficient information
training_set = pd.read_csv('train.csv').drop(['Cabin','Ticket'], axis=1)
training_set = training_set.set_index('PassengerId')

# Filling up the Embarked feature
training_set.Embarked.fillna(training_set["Embarked"].value_counts().idxmax(), inplace=True)


training_set.Age = training_set.Age.apply(lambda x: 
                                np.random.choice(training_set.Age.dropna().values)
                                if np.isnan(x) 
                                else x)

X = training_set.iloc[:,[1,3,4,5,6,7,8]].values # Omit Survived
Y = training_set.iloc[:,0].values # Survived    

# Encoding for the Gender Column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

# Encoding X categorical data + HotEncoding
from sklearn.preprocessing import  OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Avoiding Dummy Variable Trap
X = X[:,1:]

# Splitting to Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

avgs = []
stds = []
####################################################################################################
# Logistic Regression

X_train1 = X_train.copy()
X_test1 = X_test.copy()

# Feature Scaling
sc_X = StandardScaler()
X_train1 = sc_X.fit_transform(X_train1)
X_test1 = sc_X.transform(X_test1)

# Fitting Logistic Regression to Training Set
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(penalty='l2',C=0.01, solver='newton-cg',random_state=0)
classifier1.fit(X_train1, Y_train)

# Applying k-Fold Cross Validation where k=10
accuracies1 = cross_val_score(estimator=classifier1, X=X_train1, y=Y_train, cv=10)
avgs.append(accuracies1.mean()*100)
stds.append(accuracies1.std()*100)

####################################################################################################
# K-NN

X_train2 = X_train.copy()
X_test2 = X_test.copy()

# Feature Scaling
sc_X = StandardScaler()
X_train2 = sc_X.fit_transform(X_train2)
X_test2 = sc_X.transform(X_test2)

# Fitting Logistic Regression to Training Set
from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(n_neighbors=8, p=3, metric='minkowski')
classifier2.fit(X_train2, Y_train)

# Applying k-Fold Cross Validation where k=10
accuracies2 = cross_val_score(estimator=classifier2, X=X_train2, y=Y_train, cv=10)
avgs.append(accuracies2.mean()*100)
stds.append(accuracies2.std()*100)

####################################################################################################
# Kernel SVM

X_train3 = X_train.copy()
X_test3 = X_test.copy()

# Feature Scaling
sc_X = StandardScaler()
X_train3 = sc_X.fit_transform(X_train3)
X_test3 = sc_X.transform(X_test3)

# Fitting SVM Classifier to Training Set
from sklearn.svm import SVC
classifier3 = SVC(kernel='rbf', gamma=0.1, C=1)
classifier3.fit(X_train3, Y_train)

# Applying k-Fold Cross Validation where k=10
accuracies3 = cross_val_score(estimator=classifier3, X=X_train3, y=Y_train, cv=10)
avgs.append(accuracies3.mean()*100)
stds.append(accuracies3.std()*100)

####################################################################################################
# Naive Bayes

X_train4 = X_train.copy()
X_test4 = X_test.copy()

# Feature Scaling
sc_X = StandardScaler()
X_train4 = sc_X.fit_transform(X_train4)
X_test4 = sc_X.transform(X_test4)

# Fitting Naive Bayes Classifier to Training Set
from sklearn.naive_bayes import GaussianNB
classifier4 = GaussianNB()
classifier4.fit(X_train4, Y_train)

# Applying k-Fold Cross Validation where k=10
accuracies4 = cross_val_score(estimator=classifier4, X=X_train4, y=Y_train, cv=10)
avgs.append(accuracies4.mean()*100)
stds.append(accuracies4.std()*100)

####################################################################################################
# Random Forest Classification

X_train5 = X_train.copy()
X_test5 = X_test.copy()

# Feature Scaling
sc_X = StandardScaler()
X_train5 = sc_X.fit_transform(X_train5)
X_test5 = sc_X.transform(X_test5)

# Fitting Random Forest Classifier to Training Set
from sklearn.ensemble import RandomForestClassifier
classifier5 = RandomForestClassifier(n_estimators=100, criterion='entropy', max_features='auto')
classifier5.fit(X_train5, Y_train)

# Applying k-Fold Cross Validation where k=10
accuracies5 = cross_val_score(estimator=classifier5, X=X_train5, y=Y_train, cv=10)
avgs.append(accuracies5.mean()*100)
stds.append(accuracies5.std()*100)

####################################################################################################
# XGBoost

X_train6 = X_train.copy()
X_test6 = X_test.copy()

# Fitting XGBoost to the training set
from xgboost import XGBClassifier
classifier6 = XGBClassifier(max_depth=4, min_child_weight=2, colsample_bytree=0.54, subsample=0.96, gamma=0.4)
classifier6.fit(X_train6, Y_train)

# Applying k-Fold Cross Validation where k=10
accuracies6 = cross_val_score(estimator=classifier6, X=X_train6, y=Y_train, cv=10)
avgs.append(accuracies6.mean()*100)
stds.append(accuracies6.std()*100)

####################################################################################################

plt.figure(figsize=(17,6))
plt.bar(range(len(avgs)), avgs, color=(118/255,127/255,255/255))
plt.ylim([75, 85])
plt.xticks(np.arange(6),['Logistic Regression','K-NN','Kernel SVM','Naive Bayes','Random Forest','XGBoost'])
plt.grid(True)
plt.title('Comparison of all Classification models')
plt.xlabel('Models')
plt.ylabel('Accuracy Percentage')
plt.show()