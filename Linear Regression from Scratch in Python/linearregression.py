import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')

#Seperating X and y into different arrays.
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

#Test Train split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Estimate Mean and Variance
 
# Calculate the mean value of a list of numbers
def mean(values):
	return sum(values) / float(len(values))
 
# Calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar
#calculating the coefficients
def coefficients(x,y):
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]

b0, b1 = coefficients(X_train,y_train)
print('Coefficients: B0=%.3f, B1=%.3f' % (b0, b1))
y_pred = b0 + b1*X

plt.scatter(X_train,y_train,color='red')
plt.plot(X,y_pred)
plt.title('Swedish Insurance Claims(Training Set)')
plt.xlabel('number of claims')
plt.ylabel('total payment for all the claims in thousands of Swedish Kronor')
plt.show()

plt.scatter(X_test,y_test,color='red')
plt.plot(X,y_pred)
plt.title('Swedish Insurance Claims(Test Set)')
plt.xlabel('number of claims')
plt.ylabel('total payment for all the claims in thousands of Swedish Kronor')
plt.show()