import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.model_selection import cross_val_score

boston = load_boston()

boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df['price'] = boston.target

print(boston_df)


x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

"""plt.scatter(y_test,y_pred)
plt.xlabel('prices')
plt.ylabel('predicted price')
plt.show()"""


# mean square error
mse = metrics.mean_squared_error(y_test, y_pred)
print(mse)

# cross validation (K-Fold cross_val)
cv_scores = cross_val_score(reg, x, y, cv=5)
print(cv_scores)
print(np.mean(cv_scores))

# Lasso regression

lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(x, y)
lasso_coeff = lasso.coef_
print(lasso_coeff)

plt.plot(range(13), lasso_coeff)
plt.xticks(range(13), boston.feature_names)
plt.ylabel('coefficent')
plt.show()
