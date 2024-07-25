import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_california_housing
df = fetch_california_housing()

dataset=pd.DataFrame(df.data)
dataset.columns=df.feature_names
dataset.head()

## Independent features and dependent features

X=dataset
y=df.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)
# data size for testing - the part we are testing - is 30%

## standardizing the data set

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.linear_model import LinearRegression

## cross validation 

from sklearn.model_selection import cross_val_score

regression=LinearRegression()
regression.fit(X_train,y_train)

mse=cross_val_score(regression, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
# 5 diff types of accuracy - will be trained with 5 datasets 

np.mean(mse)

## prediction on the test data 

reg_predict=regression.predict(X_test)

import seaborn as sns 

sns.displot(reg_predict-y_test, kind='kde') # compare the predict and real values

# low variance - good prediction 

from sklearn.metrics import r2_score

score=r2_score(reg_predict, y_test)


