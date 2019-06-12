#####################################################################################
# Creator     : Gaurav Roy
# Date        : 11 June 2019
# Description : The code contains the Kernel Support Vector Machine model for the 
#               Titanic dataset.
#####################################################################################

# Importing Libraries
import numpy as np
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

####################################################################################################

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Applying PCA
from sklearn.decomposition import PCA

# The first 3 IVs have variance total of about 61% of total variance which is good enough
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Fitting SVM Classifier to Training Set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, Y_train)

#################################################################################################

# Applying k-Fold Cross Validation where k=10
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=10)
avg_accuracies = accuracies.mean()
std_accuracies = accuracies.std()

# Applying Grid Search to find the best model and best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[0.01,0.1,1,10], 'kernel':['linear']},
               {'C':[0.01,0.1,1,10,100,1000], 'kernel':['rbf'], 'gamma':[0.01, 0.05, 0.1, 0.2, 0.3]}]

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1)

grid_search = grid_search.fit(X_train, Y_train)
# Gives the accuracy with the best parameters
best_accuracy = grid_search.best_score_

###########################################################
# Gives the values of the best hyperparameters
best_parameters = grid_search.best_params_
###########################################################
# Best case: Without PCA, gamma=0.1, C=1, kernel=rbf

'''
Without PCA

k-fold cross accuracy = 0.825794+-0.031138
best_accuracy = 0.8286  [C=1, gamma=0.1, kernel=rbf]

PCA=3

k-fold cross accuracy = 0.769788+-0.047933
best_accuracy = 0.782  [C=10, gamma=0.1, kernel=rbf]

PCA=2

k-fold cross accuracy = 0.723443+-0.052368
best_accuracy = 0.731 [C=100, gamma=0.3, kernel=rbf]
'''