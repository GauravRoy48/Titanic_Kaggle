#####################################################################################
# Creator     : Gaurav Roy
# Date        : 12 June 2019
# Description : The code tests the best classification model on the testing set of
#               the titanic dataset. It is then stored in output.csv
#####################################################################################

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset

# Dropping Cabin and Ticket columns due to irrelevant and insufficient information
training_set = pd.read_csv('train.csv').drop(['Cabin','Ticket'], axis=1)
training_set = training_set.set_index('PassengerId')

test_set = pd.read_csv('test.csv').drop(['Cabin','Ticket'], axis=1)
test_set = test_set.set_index('PassengerId')

# training_set.isnull().any() # to check for fields containing missing data
# training_set.isnull().sum(axis=0) # to check for number of nan in fields
# test_set.isnull().sum(axis=0) # to check for number of nan in fields

# Filling up the Embarked feature
training_set.Embarked.fillna(training_set["Embarked"].value_counts().idxmax(), inplace=True)


training_set.Age = training_set.Age.apply(lambda x: 
                                np.random.choice(training_set.Age.dropna().values)
                                if np.isnan(x) 
                                else x)

test_set.Fare.fillna(test_set.Fare.mean(), inplace=True)
    
test_set.Age = test_set.Age.apply(lambda x: 
                                np.random.choice(test_set.Age.dropna().values)
                                if np.isnan(x) 
                                else x)

X_train = training_set.iloc[:,[1,3,4,5,6,7,8]].values # Omit Survived
Y_train = training_set.iloc[:,0].values # Survived


X_test = test_set.iloc[:,[0,2,3,4,5,6,7]].values 
   
fam = {'PassengerId':[1, 2, 3, 4, 5],
     'Pclass':[2, 2, 2, 2, 2],
     'Name':['Gaurav Roy', 'Mother Roy', 'Uncle Aich', 'Grandma Aich', 'Meena P.'],
     'Sex':['male', 'female', 'male', 'female', 'female'],
     'Age':[24, 49, 51, 73, 24],
     'SibSp':[1, 0, 0, 0, 1],
     'Parch':[1, 1, 1, 1, 0],
     'Fare':[32, 32, 32, 32, 32],
     'Embarked':['Q', 'Q', 'Q', 'Q', 'Q']}

fam = pd.DataFrame(data=fam)
fam = fam.set_index('PassengerId')

fam_test = fam.iloc[:,[0,2,3,4,5,6,7]].values



# Encoding for the Gender Column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train[:,1] = le.fit_transform(X_train[:,1])
X_test[:,1] = le.transform(X_test[:,1])
fam_test[:,1] = le.transform(fam_test[:,1])


# Encoding X categorical data + HotEncoding
from sklearn.preprocessing import  OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
X_train = np.array(ct.fit_transform(X_train), dtype=np.float)
X_test = np.array(ct.transform(X_test), dtype=np.float)
fam_test = np.array(ct.transform(fam_test), dtype=np.float)


# Avoiding Dummy Variable Trap
X_train = X_train[:,1:]
X_test = X_test[:,1:]
fam_test = fam_test[:,1:]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
fam_test = sc_X.transform(fam_test)

# Fitting Kernel SVM Classifier to Training Set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', gamma=0.1, C=1)
classifier.fit(X_train, Y_train)

# Predicting the Test Set Results
Y_test = classifier.predict(X_test)

temp = pd.read_csv('gender_submission.csv')

temp.Survived = Y_test

temp.to_csv('output.csv', index=None)