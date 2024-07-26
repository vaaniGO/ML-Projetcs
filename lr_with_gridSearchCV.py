import pandas as pd
import seaborn as sns
import numpy as np

df=sns.load_dataset('iris')
df.head()

df['species'].unique() #know the number of unique outputs

df.isnull().sum()

df=df[df['species']!='setosa']

df.head()   #The head() method returns a specified number of rows, string from the top. The head() method returns the first 5 rows if a number is not specified.

df['species']=df['species'].map({'versicolor':0, 'virginica':1})

X=df.iloc[:, :-1]   #integer location [row_start:row_end, column_start:column_end]
y=df.iloc[:, -1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

#test data is used for validation - for teaching the model 

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()

from sklearn.model_selection import GridSearchCV
parameter={'penalty':['l1', 'l2', 'elasticnet'], 'C':[1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50],
           'max_iter':[100, 200, 300]}

classifier_regressor=GridSearchCV(classifier, param_grid=parameter, scoring='accuracy')

classifier_regressor.fit(X_train, y_train)

print(classifier_regressor.best_params_)

print(classifier_regressor.best_score_)

##prediction
y_pred=classifier_regressor.predict(X_test)

from sklearn.metrics import accuracy_score,classification_report

score=accuracy_score(y_pred, y_test)
print(score)

print(classification_report(y_pred, y_test))

sns.pairplot(df,hue='species')
