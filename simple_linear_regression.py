import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the data from the csv file
df = pd.read_csv("FuelConsumption.csv")

# having an initial look at the dataset
df.head()

# see the summary of the data
df.describe()

# let's select some features
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

# plotting the graph for FUELCONSUMPTION_COMB against CO2EMMISIONS
# here Emission is the dependent variable while fuel consumption is the independent variable

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='black')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2EMISSIONS")
plt.show()

# we can see that the above graph is quite linear, so we 
# can use linear regression to predict the CO2EMISSIONS from FUELCONSUMPTION_COMB

# now we create a train/test split of dataset, one for training and other for testing
# this will provide more out-of-sample accuracy. Both dataset will be mutually exclusive

# we have split our dataset to train(80%) and test(20%)

msk = np.random.rand(len(df)) < 0.8 # randomly selecting data

train = cdf[msk]
test = cdf[~msk]

# Linear Regression fits a linear model with coefficients B = (B1, ..., Bn) to minimize the 'residual sum of squares' 
# between the actual value y in the dataset, and the predicted value yhat using linear approximation. 

# let's plot a graph of our train dataset

plt.scatter(train.FUELCONSUMPTION_COMB, train.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2EMISSIONS")
plt.show()

# using sklearn to model our data

from sklearn import linear_model

reg = linear_model.LinearRegression()
train_x = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
reg.fit(train_x, train_y)

# coefficients

print ('Coefficients: ', reg.coef_)
print ('Intercept: ',reg.intercept_)

# Now, plotting the fit line over the data

plt.scatter(train.FUELCONSUMPTION_COMB, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, reg.coef_[0][0]*train_x + reg.intercept_[0], '-r')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2EMISSIONS")
plt.show()
# now we check the accuracy of the model by testing it in our test data

from sklearn.metrics import r2_score

# we used errors concepts like root squared error(RSE), Mean Absolute Error, and R-squared (not an error though)
test_x = np.asanyarray(test[['FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = reg.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )