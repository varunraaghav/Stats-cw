# Imperial College London ME3 Statistics Coursework - Python Code 

# Import necessary libraries:
import math
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns
import csv
plt.style.use('seaborn')


from scipy import stats

# _______________________________________________________________________________________________________________________
# Random seed
random.seed(1847960)

# _______________________________________________________________________________________________________________________
# Importing and reading wavelength data from the designated csv. Performing prerequisite numpy operations to format data

df = pd.read_csv('vrv20.csv')


# X_list = df['X'].values.tolist()
# Y_list = df['Y'].values.tolist()

x = df['X'].values  # converting list into numpy arrays for future manipulations
y = df['Y'].values

x = x.reshape(-1, 1)  # reshaping the arrays retains the original data but changes the shape of the structure to be easily used in future operations
y = y.reshape(-1, 1)

# _______________________________________________________________________________________________________________________


# Histogram plot of the wavelngth data 

plt.figure(dpi=150)
plt.hist(y, bins=50, edgecolor='black')
plt.ylabel('Wavelength (nm)')
plt.title("Wavlength histogram")

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
std_dev_x_sample = np.std(x, ddof=1)  # calculating sample std dev instead of population std dev
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
plt.title('vrv20: Wavelength (nm) plotted against time')

# _______________________________________________________________________________________________________________________

# Calculating Linear regression y_predicted values in a for-loop, to the kth order: 
# when i = 1, the 'simple' linear fit is made. when i = 2``, the 'qudratic fit' is made, and the order of the polynomial increases when i increases
# Adding linear regression values to the scatter plot upto kth order

k = 6     # determines the max order upto which the linear regression y_predicted values are stored
linear_regression_values_df = pd.DataFrame()   # all predicted values are stored in columns in a pandas dataframe, so that the values can be easily accessed for future operations
# linear_regression_values_df['x_standardised'] = x_standardised.tolist()




log_likelihood_arr = np.zeros(k)

n = len(y)   # number of terms in the wavelength time series data. For vrv20 dataset: value = 851
for i in range (1, k+1):
    poly_k = PolynomialFeatures(degree=i, include_bias=False)
    x_poly_k = poly_k.fit_transform(x_standardised)
    model_k = LinearRegression().fit(x_poly_k, y)
    y_predicted_k = model_k.predict(x_poly_k)

    linear_regression_values_df[str(i)] = y_predicted_k.tolist()

    plt.plot(x_standardised, y_predicted_k, label='Order ' + str(i))

    #  Log likelihood calculation: SUBJECT TO BE CHANGED
    log_likelihood_arr[i-1] = (-0.5 * n * np.log(2 * np.pi * (std_dev**2))) - ( (0.5/(std_dev**2))  * np.sum((y - y_predicted_k)**2))
# ___________________________________________________________________________________________________________________________

    
plt.legend()
# plt.grid(True)

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

residuals_arr_standardised = StandardScaler().fit_transform(residuals_arr).flatten()
res_std_dev = np.std(residuals_arr_standardised)

quantiles = np.random.normal(0, res_std_dev, len(residuals_arr_standardised))

quantiles.sort()
residuals_arr_standardised.sort()

plt.figure(dpi=150)
plt.plot(quantiles, residuals_arr_standardised, '.')
plt.axline((0,0), (1,1), color='green', linestyle='--')
plt.axis('square')
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.ylabel('Residuals')
plt.xlabel('Normal Quantile')
plt.title('Residuals Q-Q plot')

# _______________________________________________________________________________________________________________________

# Plotting the scatter plot of wavelength data in indexes of value 10 (variable called delta) , plot the chosen kth order of linear regression

increments = 10  # time index incremenents
x_increments = []
y_increments = []
for i in range(increments-1, len(x_standardised),increments):
    x_increments.append(x_standardised[i])
    y_increments.append(y[i])


x_increments_array = np.array(x_increments)
y_increments_array = np.array(y_increments)


poly_k_increments = PolynomialFeatures(degree=k_selected, include_bias=False)
x_poly_k_increments = poly_k_increments.fit_transform(x_increments_array)
model_k_increments = LinearRegression().fit(x_poly_k_increments, y_increments_array)
y_predicted_k_increments = model_k_increments.predict(x_poly_k_increments)


plt.figure(dpi=150)
plt.plot(x_increments_array, y_increments_array, '.', color='green', label = 'Sample scatter')
plt.plot(x_increments_array, y_predicted_k_increments, label='Order ' + str(k_selected))
plt.ylabel('Wavelength (nm)')
plt.xlabel('Time (Standardised)')
plt.title(f'Wavelength against time (standardised) in index increments of {increments}')

# _______________________________________________________________________________________________________________________



# Bootstrapping

bootstrapped_df = pd.DataFrame()
bootstrap_iter = 10

# y_predicted_bootstrapped = y_predicted_k_increments  # before any bootstrap, y_predicted would be from model 2(f)


residuals_bootstrap = (y_increments_array - y_predicted_k_increments).flatten()

for i in range(0, bootstrap_iter):

    # residuals_bootstrap_iter_list = []

    # for j in range(0, len(residuals_bootstrap)):
    #     residuals_bootstrap_iter_list.append(random.choice(residuals_bootstrap))

    
    residuals_bootstrap_iter_arr = np.random.choice(residuals_bootstrap, len(residuals_bootstrap), replace=True)
    print(residuals_bootstrap_iter_arr)


    # residuals_bootstrap_iter_arr = np.array(residuals_bootstrap_iter_list)

    # y_response_bootstrap = y_predicted_bootstrapped + residuals_bootstrap_iter_arr

    y_response_bootstrap = y_predicted_k_increments.flatten() + residuals_bootstrap_iter_arr


    poly_bootstrap = PolynomialFeatures(degree=k_selected, include_bias=False)
    x_poly_bootstrap = poly_bootstrap.fit_transform(x_increments_array)
    model_bootstrap = LinearRegression().fit(x_poly_bootstrap, y_response_bootstrap)
    y_predicted_bootstrapped = model_bootstrap.predict(x_poly_bootstrap)


    # print(y_predicted_bootstrapped.shape)

    bootstrapped_df[str(i)] = y_predicted_bootstrapped

# print("BREAK")
# print(" ")


# def bootstrapping(actual_data, predicted_data, iterations):
#     bootstrapped_df = pd.DataFrame()
#     residuals_bootstrap = (actual_data - predicted_data).flatten()

#     for i in range(0, iterations):

#         residuals_bootstrap_iteration = np.random.choice(residuals_bootstrap, len(residuals_bootstrap), replace=True)
#         print(residuals_bootstrap_iteration)

#         y_response_bootstrap = predicted_data.flatten() + residuals_bootstrap_iteration

#         y_predicted_bootstrapped = linear_regression(x_increments_array, y_response_bootstrap, k_selected)

#         bootstrapped_df[str(i)] = y_predicted_bootstrapped

#     return bootstrapped_df


# def linear_regression(x_data, y_data, k):
#     poly = PolynomialFeatures(degree=k, include_bias=False)
#     x_poly = poly.fit_transform(x_data)
#     model = LinearRegression().fit(x_poly, y_data)
#     y_predicted = model.predict(x_poly)

#     return y_predicted

# bootstrapped_df2 = bootstrapping(y_increments_array, y_predicted_k_increments, 10)











quantile_025 = np.zeros(len(bootstrapped_df))
quantile_975 = np.zeros(len(bootstrapped_df))
for i in range(0, len(bootstrapped_df)):

    quantile_array = bootstrapped_df.iloc[i].values
    quantile_025[i] =  np.quantile(quantile_array, 0.025)
    quantile_975[i] = np.quantile(quantile_array, 0.975)

plt.plot(x_increments_array, quantile_025, linestyle='--', color='orange', label='0.025 quantile')
plt.plot(x_increments_array, quantile_975, linestyle='--', color='red', label='0.975 quantile')
plt.legend()


print(len(quantile_025))
print(" ")
print(len(quantile_975))
print(" ")
print(len(x_increments_array))   





# f = open('trial.csv', 'w')
# writer = csv.writer(f)
# writer.writerow(x)
# writer.writerow(x_standardised)
# writer.writerow(y)
# writer.writerow(x_increments_array)
# writer.writerow(y_increments_array)
# f.close()




# Show the necessary graphs at the end (commented out once the code is assumed to be working)
# plt.show()

