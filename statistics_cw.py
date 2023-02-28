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

# ___________________________________________________________________

# Random seed
random.seed(1847960)
# ___________________________________________________________________

# Importing and reading wavelength data from the designated csv
df = pd.read_csv('vrv20.csv')


X_list = df['X'].values.tolist()
Y_list = df['Y'].values.tolist()

x = np.array(X_list)  # converting list into numpy arrays for future manipulations
y = np.array(Y_list)

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



# Present Statistic values in a table form (using pd table)

mean = np.mean(y)
trimmed_mean = stats.trim_mean(y, 0.1)
median = np.median(y)
std_dev = np.std(y)
q1, q3 = np.percentile(y, [25, 75])
iqr = q3 - q1

# Create a table with the statistics
table_data = {
    'Statistic': ['Mean', '10% Trimmed Mean', 'Median', 'Standard Deviation', 'Interquartile Range'],
    'Value': [mean, trimmed_mean, median, std_dev, iqr]
}
table = pd.DataFrame(table_data)
print(table)


# Simple linear regression model of the wavelngth data

poly_simple = PolynomialFeatures(degree=1, include_bias=False)
x_reshaped = x.reshape(-1, 1)
y_reshaped = y.reshape(-1, 1)

x_poly_simple = poly_simple.fit_transform(x_reshaped)
model_simple = LinearRegression().fit(x_poly_simple, y_reshaped)
y_predicted_simple = model_simple.predict(x_poly_simple)

# linear regression with polynomial term (ie k = 2)

poly_2 = PolynomialFeatures(degree=2, include_bias=False)

x_poly_2 = poly_2.fit_transform(x_reshaped)
model_2 = LinearRegression().fit(x_poly_2, y_reshaped)
y_predicted_2 = model_2.predict(x_poly_2)


# scatter plot of the wavelength data , with the linear regression values (exercise 2)
plt.figure(dpi=150)
plt.plot(x, y, '.')
plt.ylabel('Wavelength (nm)')
plt.xlabel('Time')
plt.title('vrv20: Wavlength (nm) plotted against time')

plt.plot(x_reshaped, y_predicted_simple, color = 'red', label='k=1 (Simple) linear regression model') #simple linear regression
plt.plot(x_reshaped, y_predicted_2, color = 'green', label='k = 2 (qudratic) linear regression model') #quadrating linear regression


plt.legend()
plt.grid(True)


# Show the necessary graphs at the end (commented out once the code is assumed to be working)
plt.show()