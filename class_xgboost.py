#####################################################################################
# Creator     : Gaurav Roy
# Date        : 11 June 2019
# Description : The code contains the XGBoost model for the Titanic dataset.
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

# Applying PCA
from sklearn.decomposition import PCA

# The first 3 IVs have variance total of about 61% of total variance which is good enough
#pca = PCA(n_components=3)
#X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)

# Fitting XGBoost to the training set
from xgboost import XGBClassifier
classifier = XGBClassifier(max_depth=5, min_child_weight=7, colsample_bytree=0.9, subsample=0.9)
classifier.fit(X_train, Y_train)

#################################################################################################

# Applying k-Fold Cross Validation where k=10
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=25)
avg_accuracies = accuracies.mean()
std_accuracies = accuracies.std()

# Applying Grid Search to find the best model and best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'max_depth':[4],
               'min_child_weight':[2],
               'gamma':[0.4],
               'colsample_bytree':[0.54],
               'subsample':[0.96]}]

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
# Best case: Without PCA, criterion=entropy, max_features=auto, n_estimators=100

'''
Without PCA

k-fold cross accuracy = 0.775362+-0.028775
best_accuracy = 0.8047 [max_depth=6, max_features=auto, n_estimators=100]
'''