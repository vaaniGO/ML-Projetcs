import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df=pd.read_csv('Ecommerce Customers')
df.head()

sns.jointplot(x='Time on App', y='Yearly Amount Spent', 
              data=df, alpha=0.5)

sns.pairplot(df, kind='scatter', plot_kws={'alpha':0.4})

X=df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

y=df.iloc[:, -1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# if you ever want to reporduce the same split, pass the same data and the same random number

# creating and training the model

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train, y_train)

# lets see how important each of the variables is to our model

cdf=pd.DataFrame(lr.coef_, X.columns, columns=['coeff'])
print(cdf)

# creating some predictions 

predictions=lr.predict(X_test)

# plot the predicted vs actual values to see accuracy 

sns.scatterplot([predictions, y_test])
plt.xlabel('Predictions')

from sklearn.metrics import mean_squared_error, mean_absolute_error

print('MAE: ', mean_absolute_error(y_test, predictions))
print('MSE: ', mean_squared_error(y_test, predictions))

# residual analysis 

residuals=y_test-predictions
sns.displot(residuals)

import pylab
import scipy.stats as stats

stats.probplot(residuals, dist='norm', plot=pylab)
pylab.show()