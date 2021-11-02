import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# reading the data

df = pd.read_csv("FuelConsumption.csv")

df.head()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

# plotting a graph between enginesize and co2emissions
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# creating test/train sets in 80:20 ratio 
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# using multiple regression model from sklearn
# in multiple regression more than one independent variables are present to predict one dependent variable

from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_) # Coefficient and Intercept, are the parameters of the fit line

# we can use ordinary least square method it tries to minimizes 
# the sum of squared errors (SSE) or mean squared error (MSE) between the target variable (y) and our predicted output

# prediction
y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# greater is the variance, more is the accuracy
print('Variance score: %.2f' % regr.score(x, y))