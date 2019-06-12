#####################################################################################
# Creator     : Gaurav Roy
# Date        : 11 June 2019
# Description : The code contains the k-nearest neighbours model for the Titanic
#               dataset.
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
pca = PCA(n_components=3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Fitting K-NN Classifier to Training Set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
classifier.fit(X_train, Y_train)

#################################################################################################

# Applying k-Fold Cross Validation where k=10
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=10)
avg_accuracies = accuracies.mean()
std_accuracies = accuracies.std()

# Applying Grid Search to find the best model and best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_neighbors':[3,5,8,10], 'weights':['uniform','distance'], 'p':[2], 'metric':['euclidean']},
               {'n_neighbors':[3,5,8,10], 'weights':['uniform','distance'], 'p':[1], 'metric':['manhattan']},
               {'n_neighbors':[3,5,8,10], 'weights':['uniform','distance'], 'p':[3,4,5,6], 'metric':['minkowski']}]

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
# Best case: Without PCA, metric=minkowski, n_neighbours=8, p=3, weights=uniform

'''
Without PCA

k-fold cross accuracy = 0.789409+-0.040066
best_accuracy = 0.817  [metric=minkowski, n_neighbours=8, p=3, weights=uniform]


PCA=3

k-fold cross accuracy = 0.752847+-0.057878
best_accuracy = 0.764  [metric=euclidean, n_neighbours=8, p=1, weights=uniform]


PCA=2

k-fold cross accuracy = 0.716342+-0.036417
best_accuracy = 0.7275 [metric=manhattan, n_neighbours=5, p=2, weights=uniform]
'''