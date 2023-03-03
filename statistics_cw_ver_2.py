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
from scipy import stats

plt.style.use('seaborn')


# Random seed
random.seed(1847960)


# Importing and reading wavelength data from the designated csv. Performing prerequisite numpy operations to format data
df = pd.read_csv('vrv20.csv')


x = df['X'].values  # converting list into numpy arrays for future manipulations
y = df['Y'].values

x = x.reshape(-1, 1)  # reshaping the arrays retains the original data but changes the shape of the structure to be easily used in future operations
y = y.reshape(-1, 1)

# _____________________________________________________________________________________________________________________________________________


def histogram(data, bin_number):
    plt.figure(dpi=150)
    plt.hist(data, bins=bin_number, edgecolor='black')
    plt.ylabel('Wavelength (nm)')
    plt.xlabel('Time index')
    plt.title(f"Wavlength histogram. Number of bins: {bin_number}")



def boxplot(data):
    plt.figure(dpi=150)
    plt.boxplot(data)
    plt.ylabel('Wavelength (nm)')
    plt.title("Wavlength Boxplot")



def general_statistics(data):
    mean = np.mean(y)
    trimmed_mean = stats.trim_mean(y, 0.1)
    median = np.median(y)
    std_dev = np.std(y, ddof=0)      # This is the population standard deviation, different from sample standard deviation
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1

    table_data = {
    'Statistic': ['Mean', '10% Trimmed Mean', 'Median', 'Standard Deviation', 'Interquartile Range'],
    'Value': [mean, trimmed_mean, median, std_dev, iqr]
    }
    
    table = pd.DataFrame(table_data)
    return table , std_dev



def scatter_function(x_data, y_data, color_value):
    plt.figure(dpi=150)
    plt.plot(x_data, y_data, '.', color = color_value)
    plt.ylabel('Wavelength (nm)')
    plt.xlabel('Time')



# histogram(y, 30)
# boxplot(y)

stats_table , std_dev = general_statistics(y)
print(stats_table)



# _____________________________________________________________________________________________________________________________________________

# Question 2:

def linear_regression(x_data, y_data, k):
    poly = PolynomialFeatures(degree=k, include_bias=False)
    x_poly = poly.fit_transform(x_data)
    model = LinearRegression().fit(x_poly, y_data)
    y_predicted = model.predict(x_poly)

    return y_predicted



scatter_function(x,y,"green")
plt.title("UNSTANDARDISED: Wavlength (nm) plotted against time")
y_predicted_simple = linear_regression(x,y,1)
plt.plot(x, y_predicted_simple, label='simple linear regression')
y_predicted_quadratic = linear_regression(x,y,2)
plt.plot(x, y_predicted_quadratic, label='Quadratic fit', color='orange')
plt.legend()

# _____________________________________________________________________________________________________________________________________________


def standardisation(data):
    mean = np.mean(data)
    std_dev = np.std(data, ddof = 1)
    standardised_data = (data - mean) / std_dev

    return standardised_data

x_standardised = standardisation(x)

scatter_function(x_standardised,y,'blue')
plt.title('Standardised wavelength plot')

# _____________________________________________________________________________________________________________________________________________


def log_likelihood(actual_data, predicted_data, std_dev):
    n = len(actual_data)

    value = (-0.5 * n * np.log(2 * np.pi * (std_dev**2))) - ( (0.5/(std_dev**2))  * np.sum((actual_data - predicted_data)**2))
    
    return value


k_max = 6
linear_regression_values_df = pd.DataFrame()

log_likelihood_arr = np.zeros(k_max)

for i in range(1, k_max+1):
    y_temp = linear_regression(x_standardised, y, i)
    plt.plot(x_standardised, y_temp, label='Order ' + str(i))
    
    log_likelihood_arr[i-1] = log_likelihood(y, y_temp, std_dev)
    
    linear_regression_values_df[str(i)] = y_temp.tolist()

plt.legend()

# _____________________________________________________________________________________________________________________________________________


def AIC(log_data, k_max):
    n = len(log_data)
    AIC_arr = np.zeros(n)

    for i in range(0,n):
        q = i+2  # 'q' = number of parameters, which equals order + 1. Order array stars at 0th index, therefore q = i+2
        AIC_arr[i] =  2*(q) - 2*log_data[i]


    AIC_table = pd.DataFrame()

    AIC_table['Linear regression Order (ie parameters)'] = np.arange(1, k_max+1, 1)
    AIC_table['Max Log likelihood'] = log_data
    AIC_table['AIC'] = AIC_arr

    min_AIC_k = 1+np.argmin(AIC_arr)

    return AIC_table , min_AIC_k


AIC_table , k_selected = AIC(log_likelihood_arr, k_max)
print(AIC_table)
print("Chosen model for linear regression based on lowest AIC value is of order: ", k_selected)

# _____________________________________________________________________________________________________________________________________________

def residual(actual_data, predicted_data):
    residual_arr = actual_data - predicted_data
    return residual_arr

y_pred_chosen_order = np.array(linear_regression_values_df[str(k_selected)].values.tolist())
residuals = residual(y, y_pred_chosen_order)

residuals_arr_standardised = standardisation(residuals).flatten()
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

# _____________________________________________________________________________________________________________________________________________


def sampling(data, increments):
    value_list = []
    for i in range(increments-1, len(data), increments):
        value_list.append(data[i])

    return value_list



x_increments_array = np.array(sampling(x_standardised, 10))
y_increments_array = np.array(sampling(y, 10))


y_predicted_k_increments = linear_regression(x_increments_array, y_increments_array, k_selected)


scatter_function(x_increments_array, y_increments_array, 'orangered')
plt.title('Wavelength against time (standardised) in index increments of 10')
plt.plot(x_increments_array, y_predicted_k_increments, label='Order ' + str(k_selected))


# _____________________________________________________________________________________________________________________________________________

def bootstrapping(actual_data, predicted_data, iterations):
    bootstrapped_df = pd.DataFrame()
    residuals_bootstrap = (actual_data - predicted_data).flatten()

    for i in range(0, iterations):

        residuals_bootstrap_iteration = np.random.choice(residuals_bootstrap, len(residuals_bootstrap), replace=True)

        y_response_bootstrap = predicted_data.flatten() + residuals_bootstrap_iteration

        y_predicted_bootstrapped = linear_regression(x_increments_array, y_response_bootstrap, k_selected)

        bootstrapped_df[str(i)] = y_predicted_bootstrapped

    return bootstrapped_df





def quantile_function(data, quantile_number):    # data is given as a dataframe here
    quantile_array = np.zeros(len(data))
    
    for i in range(0, len(data)):
        quantile_data = data.iloc[i].values
        
        quantile_array[i] = np.quantile(quantile_data, quantile_number)

    return quantile_array




bootstrap_iter = 4
bootstrapped_df = bootstrapping(y_increments_array, y_predicted_k_increments, bootstrap_iter)

quantile_025 = quantile_function(bootstrapped_df, 0.025)
quantile_975 = quantile_function(bootstrapped_df, 0.975)

plt.fill_between(x_increments_array.squeeze(), quantile_025, quantile_975, alpha = 0.3, label=f'95% Confidence band with {bootstrap_iter} iterations')
plt.legend()

# _____________________________________________________________________________________________________________________________________________


# bootstrapping iterations = 2,10,100,500,3000

bootstrap_multiple_iter = [20, 100, 500, 3000]

fig,ax = plt.subplots(ncols=len(bootstrap_multiple_iter), dpi=150)
fig.tight_layout()
fig.suptitle('Subplots showing confidence interval convergence as bootstrap sample increases')

for i in range(0, len(bootstrap_multiple_iter)):
    bootstrapped_df = bootstrapping(y_increments_array, y_predicted_k_increments, bootstrap_multiple_iter[i])

    quantile_025 = quantile_function(bootstrapped_df, 0.025)
    quantile_975 = quantile_function(bootstrapped_df, 0.975)

    ax[i].plot(x_increments_array, y_increments_array, '.')
    # ax[i].ylabel('Wavelength (nm)')
    # ax[i].xlabel('Time')
    ax[i].plot(x_increments_array, y_predicted_k_increments)

    ax[i].fill_between(x_increments_array.squeeze(), quantile_025, quantile_975, alpha = 0.3, label=f'95% Confidence band with {bootstrap_multiple_iter[i]} iterations')


# _____________________________________________________________________________________________________________________________________________

plt.show()


