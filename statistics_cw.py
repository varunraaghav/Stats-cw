# Imperial College London ME3 Statistics Coursework - Python Code 

# Import necessary libraries:
import math
import numpy as np
import pandas as pd
import random
# import scipy
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from scipy import stats

# _______________________________________________________________________________________________________________________
# Random seed
random.seed(1847960)
# _______________________________________________________________________________________________________________________
# Importing and reading wavelength data from the designated csv. Performing prerequisite numpy operations to format data

df = pd.read_csv('vrv20.csv')


X_list = df['X'].values.tolist()
Y_list = df['Y'].values.tolist()

x = np.array(X_list)  # converting list into numpy arrays for future manipulations
y = np.array(Y_list)

x = x.reshape(-1, 1)  # reshaping the arrays retains the original data but changes the shape of the structure to be easily used in future operations
y = y.reshape(-1, 1)

# _______________________________________________________________________________________________________________________


# Histogram plot of the wavelngth data 

# plt.figure(dpi=150)
# plt.hist(y)
# plt.ylabel('Wavelength (nm)')
# plt.title("Wavlength histogram")

# Box plot of the wavelength data

# plt.figure(dpi=150)
# plt.boxplot(y)
# plt.ylabel('Wavelength (nm)')
# plt.title("Wavlength Boxplot")

# _______________________________________________________________________________________________________________________



# Present Statistic values in a table form (using pd table)

mean = np.mean(y)
trimmed_mean = stats.trim_mean(y, 0.1)
median = np.median(y)
std_dev = np.std(y, ddof=0)      # This is the population standard deviation, different from sample standard deviation
q1, q3 = np.percentile(y, [25, 75])
iqr = q3 - q1

# Create a table with the statistics
table_data = {
    'Statistic': ['Mean', '10% Trimmed Mean', 'Median', 'Standard Deviation', 'Interquartile Range'],
    'Value': [mean, trimmed_mean, median, std_dev, iqr]
}
table = pd.DataFrame(table_data)
print(table)
# _____________________________________________________________________________________________________________________



# Plotting Unstandardised Scatter plot for Wavelength vs time: USE IN REPORT
# 
# plt.figure(dpi=150)
# plt.plot(x, y, '.', color = 'green')
# plt.ylabel('Wavelength (nm)')
# plt.xlabel('Time')
# plt.title('UNSTANDARDISED: Wavlength (nm) plotted against time')
# plt.grid(True)
# 
# _______________________________________________________________________________________________________________________



# Standardising the time index: X

mean_x = np.mean(x)
std_dev_x_sample = np.std(x, ddof=1)  # calculating sample mean instead of population mean
x_standardised = (x - mean_x) / std_dev_x_sample

# _____________________________________________________________________________________________________________________



# # Simple linear regression model of the wavelength data

# poly_simple = PolynomialFeatures(degree=1, include_bias=False)

# x_poly_simple = poly_simple.fit_transform(x)
# model_simple = LinearRegression().fit(x_poly_simple, y)
# y_predicted_simple = model_simple.predict(x_poly_simple)

# # _______________________________________________________________________________________________________________________




# # linear regression with polynomial term (ie k = 2)

poly_2 = PolynomialFeatures(degree=2, include_bias=False)

x_poly_2 = poly_2.fit_transform(x)
model_2 = LinearRegression().fit(x_poly_2, y)
y_predicted_2 = model_2.predict(x_poly_2)

# _______________________________________________________________________________________________________________________


# scatter plot of the wavelength data , with the linear regression values (exercise 2)
plt.figure(dpi=150)
plt.plot(x_standardised, y, '.')
plt.ylabel('Wavelength (nm)')
plt.xlabel('Time (Standardised)')
plt.title('vrv20: Wavlength (nm) plotted against time')



# Calculating Linear regression y_predicted values in a for-loop, to the kth order: 
# when i = 1, the 'simple' linear fit is made. when i = 2, the 'qudratic fit' is made, and the order of the polynomial increases when i increases

k = 5     # determines the max order upto which the linear regression y_predicted values are stored
linear_regression_values_df = pd.DataFrame()   # all predicted values are stored in columns in a pandas dataframe, so that the values can be easily accessed for future operations
linear_regression_values_df['x_standardised'] = x_standardised.tolist()


for i in range (1, k+1):
    poly_k = PolynomialFeatures(degree=i, include_bias=False)
    x_poly_k = poly_k.fit_transform(x)
    model_k = LinearRegression().fit(x_poly_k, y)
    y_predicted_k = model_k.predict(x_poly_k)

    linear_regression_values_df[str(i)] = y_predicted_k.tolist()

    plt.plot(x_standardised, y_predicted_k, label='Order ' + str(i))


plt.legend()
plt.grid(True)

# _______________________________________________________________________________________________________________________


# Show the necessary graphs at the end (commented out once the code is assumed to be working)
plt.show()

