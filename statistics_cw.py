# Imperial College London ME3 Statistics Coursework - Python Code 

# Import necessary libraries:
import math
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns
plt.style.use('seaborn')


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

# # Box plot of the wavelength data

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

# poly_2 = PolynomialFeatures(degree=2, include_bias=False)

# x_poly_2 = poly_2.fit_transform(x)
# model_2 = LinearRegression().fit(x_poly_2, y)
# y_predicted_2 = model_2.predict(x_poly_2)

# _______________________________________________________________________________________________________________________


# scatter plot of the wavelength data 
plt.figure(dpi=150)
plt.plot(x_standardised, y, '.')
plt.ylabel('Wavelength (nm)')
plt.xlabel('Time (Standardised)')
plt.title('vrv20: Wavlength (nm) plotted against time')

# _______________________________________________________________________________________________________________________

# Calculating Linear regression y_predicted values in a for-loop, to the kth order: 
# when i = 1, the 'simple' linear fit is made. when i = 2``, the 'qudratic fit' is made, and the order of the polynomial increases when i increases
# Adding linear regression values to the scatter plot upto kth order

k = 6     # determines the max order upto which the linear regression y_predicted values are stored
linear_regression_values_df = pd.DataFrame()   # all predicted values are stored in columns in a pandas dataframe, so that the values can be easily accessed for future operations
linear_regression_values_df['x_standardised'] = x_standardised.tolist()

log_likelihood_arr = np.zeros(k)

for i in range (1, k+1):
    poly_k = PolynomialFeatures(degree=i, include_bias=False)
    x_poly_k = poly_k.fit_transform(x)
    model_k = LinearRegression().fit(x_poly_k, y)
    y_predicted_k = model_k.predict(x_poly_k)

    linear_regression_values_df[str(i)] = y_predicted_k.tolist()

    plt.plot(x_standardised, y_predicted_k, label='Order ' + str(i))

    #  Log likelihood calculation: SUBJECT TO BE CHANGED
    log_likelihood_arr[i-1] = (-0.5 * 851 * np.log(2 * np.pi * (std_dev**2))) - ( (0.5/(std_dev**2))  * np.sum((y - y_predicted_k)**2))
    # ___________________________________________________________________________________________________________________________

    
plt.legend()
plt.grid(True)

# _______________________________________________________________________________________________________________________


# AIC Criterion:

AIC_arr = np.zeros(len(log_likelihood_arr))
for i in range(0, len(log_likelihood_arr)):
    q = i+2  # 'q' = number of parameters, which equals order + 1. Order array stars at 0th index, therefore q = i+2
    AIC_arr[i] =  2*(q) - 2*log_likelihood_arr[i]

AIC_table = pd.DataFrame()

AIC_table['Linear regression Order (ie parameters)'] = np.arange(1, k+1, 1)
AIC_table['Max Log likelihood'] = log_likelihood_arr
AIC_table['AIC'] = AIC_arr


print(AIC_table)
k_selected = 1+np.argmin(AIC_arr)
print("Chosen model for linear regression based on lowest AIC value is of order: ", k_selected)

# _______________________________________________________________________________________________________________________

# Calculating residuals for the chosen k order:

y_pred_chosen_order = np.array(linear_regression_values_df[str(k_selected)].values.tolist())
residuals_arr = y - y_pred_chosen_order

plt.figure(dpi=150)
plt.plot(x_standardised, residuals_arr, '.', color = 'orange')
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Residual Plot')



plt.figure(dpi=150)

plt.ylabel('Wavelength (nm)')
plt.xlabel('Time (Standardised)')

plt.plot(x_standardised, y, '.')
plt.plot(x_standardised, y_pred_chosen_order)

for i in range(len(x_standardised)):
    plt.plot([x_standardised[i], x_standardised[i]], [y[i], y_pred_chosen_order[i]], color='gray', linestyle='--')


# Show the necessary graphs at the end (commented out once the code is assumed to be working)
# plt.show()

