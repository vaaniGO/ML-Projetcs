# All techniques of hyper parameter optimization: GridSearchCV, RandomizedSearchCV, Bayesian optimization,
# Sequential model-based optimization, Optuna-Automate Hyperparameter tuning, Genetic Algorithms 
# Random forest classifier has many parameters to play with 

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# loading the data from sklearn 

breast_cancer_dataset=sklearn.datasets.load_breast_cancer()

data_frame=pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)

# adding the target column to the dataset which contains either 0 or 1 for benign or malignant 

data_frame['label']=breast_cancer_dataset.target

data_frame.shape # check the no. of rows and cols in the dataset 

data_frame.isnull().sum() # check for missing values 

data_frame['label'].value_counts() # check the no. of 1s and 0s i.e. the distribution of target variables

X=data_frame.drop(columns='label', axis=1)
y=data_frame['label']

# create numpy array for easier processing -> earlier was pandas dataframa
X=np.asarray(X)
y=np.asarray(y)

# GridSearchCV is used for determining the best hyperparameter combination for an ML model

# loading the SVC model 

model=SVC()

# create a dictionary with hyperparameter values 

parameters={
    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
    'C':[1, 5, 10, 20]
}

# grid search 

classifier=RandomizedSearchCV(model, parameters, cv=5)    # model is SVC 
# cv is cross validation, can be 3, 10 etc. here, mean of 5 folds of the data will be the final accuracy value 

# fitting the data to our model 

classifier.fit(X, y)

classifier.cv_results_ # prints the results of the Grid Search 

# results : C=10, kernel='linear'
# lets load the results to a pandas dataframe 

results=pd.DataFrame(classifier.cv_results_)

results.head()  # 5 splits because cv was given as 5 

highest_accuracy=classifier.best_score_

#95.2% is the highest accuracy we are getting (0.952... is the output)

grid_search_result=results[['param_C', 'param_kernel', 'mean_test_score']]
