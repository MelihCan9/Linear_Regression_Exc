import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

customers = pd.read_csv('Ecommerce Customers')
# print(customers.head())
print(customers.info())
#print(customers.describe())

X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)

cfd = pd.DataFrame(lm.coef_, X_train.columns, columns=['Coefficients'])
print(cfd)

predictions = lm.predict(X_test)

plt.scatter(y_test, predictions), plt.show()

from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

print("MAE: ", mae, " MSE: ", mse, " RMSE: ", rmse)

plt.hist(predictions), plt.show()

from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()

