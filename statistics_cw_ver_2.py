# Imperial College London ME3 Statistics Coursework - Python Code 

# Import necessary libraries:
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

# using seaborn style plots
plt.style.use('seaborn')


# Random seed
random.seed(1847960)


# Importing and reading wavelength data from the designated csv. Performing prerequisite numpy operations to format data
df = pd.read_csv('vrv20.csv')

# converting list into numpy arrays for future manipulations
x = df['X'].values  # Time values  
y = df['Y'].values  # wavelength values

# reshaping the arrays retains the original data but changes the shape of the structure to be easily used in future operations
x = x.reshape(-1, 1)  
y = y.reshape(-1, 1)

# _____________________________________________________________________________________________________________________________________________

# Question 1(a): Plotting a histogram and boxplot for the wavelength data

def histogram(data, bin_number):
    plt.figure(dpi=150)
    plt.hist(data, bins=bin_number, edgecolor='black')   # plt.hist function is used
    plt.ylabel('Wavelength (nm)')
    plt.xlabel('Time index')
    plt.title(f"Wavlength histogram. Number of bins: {bin_number}")



def boxplot(data):
    plt.figure(dpi=150)
    plt.boxplot(data)                   # plt.boxplot function is used
    plt.ylabel('Wavelength (nm)')
    plt.title("Wavelength Boxplot")


# Question 1(b): Getting general statistics of the wavelngth data

def general_statistics(data):
    mean = np.mean(y)
    trimmed_mean = stats.trim_mean(y, 0.1)   #10% trimmed mean
    median = np.median(y)
    std_dev = np.std(y, ddof=0)      # This is the population standard deviation (since ddof = 0), different from sample standard deviation 
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1


    table_data = {
    'Statistic': ['Mean', '10% Trimmed Mean', 'Median', 'Standard Deviation', 'Interquartile Range'],
    'Value': [mean, trimmed_mean, median, std_dev, iqr]
    }
    
    table = pd.DataFrame(table_data)  # putting the statistics into a pandas datafram

    return table , std_dev   # returning the table dataframe, and the standard deviation of wavelength as it used later on


# Question 1(c): Making a function to plot the wavelength against time data

def scatter_function(x_data, y_data, color_value):
    plt.figure(dpi=150)
    plt.plot(x_data, y_data, '.', color = color_value)
    plt.ylabel('Wavelength (nm)')
    plt.xlabel('Time')



# Calling above functions for question 1
histogram(y, 30)  # histogram bins set to 30
boxplot(y)

stats_table , std_dev = general_statistics(y)
print(stats_table)
print(" ")


# _____________________________________________________________________________________________________________________________________________

# QUESTION 2

# Question 2(a): defining a linear_regression function to output the predicted y values based in inputted x_data , y_data and the desired order 'k'

def linear_regression(x_data, y_data, k):
    poly = PolynomialFeatures(degree=k, include_bias=False)   # using the required PolynomialFeatures function
    x_poly = poly.fit_transform(x_data)
    model = LinearRegression().fit(x_poly, y_data)
    y_predicted = model.predict(x_poly)     # outputting a numpy array of predicted values

    return y_predicted



scatter_function(x,y,"green")   # CONTINUATION OF 1(C): Calling the scatter function to plot the wavelength against (unstandardised) time data
plt.title("Wavlength (nm) plotted against time (unstandardised)")

y_predicted_simple = linear_regression(x,y,1)  # 2(a) - calling linear_regression function to plot linear fit on scatter plot. Therefore order k=1 is inputted into function
plt.plot(x, y_predicted_simple, label='Simple Linear Regression')

y_predicted_quadratic = linear_regression(x,y,2) # 2(b) - calling linear_regression function to plot quadratic fit on scatter plot. Therefore order k=2 is inputted into function
plt.plot(x, y_predicted_quadratic, label='Quadratic Fit', color='orange')
plt.legend()

# _____________________________________________________________________________________________________________________________________________

# Question 2(c): Standardising the time index using the given formula in question booklet

def standardisation(data):
    mean = np.mean(data)
    std_dev = np.std(data, ddof = 1)   # Sample standard deviation is used, therefore ddof=1 is set
    standardised_data = (data - mean) / std_dev  # standardised data returned in the form of numpy array

    return standardised_data

x_standardised = standardisation(x)    # calling standardisation function to standardise the time index

scatter_function(x_standardised,y,'dimgrey')  # making a new scatter plot for wavelength against STANDARDISED time data
plt.title('Standardised Wavelength plot')

# _____________________________________________________________________________________________________________________________________________

# Continuation of 2(c) and 2(d): Fitting higher order polynomials, calculating maximum log-likelihood and getting the AIC criterion

# Log-likelihood function
def log_likelihood(actual_data, predicted_data, std_dev):
    n = len(actual_data)

    value = (-0.5 * n * np.log(2 * np.pi * (std_dev**2))) - ( (0.5/(std_dev**2))  * np.sum((actual_data - predicted_data)**2))   # formula obtained from page 150 of ME3 Stats notes

    return value


k_max = 8  # setting the maximum order for which linear regression will be performed
linear_regression_values_df = pd.DataFrame()

log_likelihood_arr = np.zeros(k_max)

for i in range(1, k_max+1):
    
    y_temp = linear_regression(x_standardised, y, i)  # finding the predicted y values for that specific order, starting from k=1, by calling the linear_regression func
    plt.plot(x_standardised, y_temp, label='Order ' + str(i))   # standardised time is used
    
    log_likelihood_arr[i-1] = log_likelihood(y, y_temp, std_dev)  # storing the max log-likelihood for the regression order
    
    linear_regression_values_df[str(i)] = y_temp.tolist()   # storing the predicted y values into a new column in a pandas dataframe

plt.legend()

# _____________________________________________________________________________________________________________________________________________

# AIC 

def AIC(log_data, k_max):
    n = len(log_data)
    AIC_arr = np.zeros(n)

    for i in range(0,n):
        q = i+2  # 'q' = number of parameters, which equals (order + 1). Order array stars at 0th index, therefore q = i+2
        AIC_arr[i] =  2*(q) - 2*log_data[i]   # calculating the AIC for given log likelihood data and parameters 


    AIC_table = pd.DataFrame() # storing the AIC data in a pandas dataframe

    AIC_table['Linear regression Order (ie parameters)'] = np.arange(1, k_max+1, 1)
    AIC_table['Max Log likelihood'] = log_data
    AIC_table['AIC'] = AIC_arr

    min_AIC_k = 1+np.argmin(AIC_arr)   # finding the index for which the AIC value is the lowest. Since python index starts at 0, order selected = index + 1

    return AIC_table , min_AIC_k


AIC_table , k_selected = AIC(log_likelihood_arr, k_max)
print(AIC_table)
print(" ")
print("Chosen model for linear regression based on lowest AIC value is of order: ", k_selected)

# _____________________________________________________________________________________________________________________________________________

# Question 2(e): Calculating the residuals for the chosen order from 2(d) - ie based on k_selected

def residual(actual_data, predicted_data):
    residual_arr = actual_data - predicted_data
    return residual_arr

y_pred_chosen_order = np.array(linear_regression_values_df[str(k_selected)].values.tolist())  # getting the values from 2(c) which was stored in a dataframe and pytting into a numpy array

residuals = residual(y, y_pred_chosen_order)  # calculating the residuals for that order

residuals_arr_standardised = standardisation(residuals).flatten()   # standardising the residuals data
res_std_dev = np.std(residuals_arr_standardised)  # finding the population std deviation of the standardised residual data

quantiles = np.random.normal(0, res_std_dev, len(residuals_arr_standardised))   # finding the normal distribution of that residual_data using mean = 0 (since residual_data is standardised) and standard deviation = res_std_dev

# sorting the normal distribution data, and the standardised residual data (both are numpy array)
quantiles.sort()
residuals_arr_standardised.sort()


plt.figure(dpi=150)
plt.plot(quantiles, residuals_arr_standardised, '.')  # plotting the sorted quantiles on the x axis, and the residuals on the y_axis to make a residual q-q plot
plt.axline((0,0), (1,1), color='green', linestyle='--')  # plotting a y=x line to see the deviation from a normal distribution
plt.axis('square')
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.ylabel('Residuals')
plt.xlabel('Normal Quantile')
plt.title('Residuals Q-Q plot')

# _____________________________________________________________________________________________________________________________________________

# Question 2(f): Getting samples from the entire wavelength dataset in iterations of 10

# sampling function to return the sampled_data baaed on the increments set
def sampling(data, increments):
    value_list = []
    for i in range(increments-1, len(data), increments):
        value_list.append(data[i])

    return value_list



x_increments_array = np.array(sampling(x_standardised, 10))   # increments set to 10
y_increments_array = np.array(sampling(y, 10))

# calculating the y_predicted values for the sampled data using the linear_regression function, of order 'k_selected' based on the lowest AIC value
y_predicted_k_increments = linear_regression(x_increments_array, y_increments_array, k_selected)


# plotting the sampled x,y data and the linear regression y_pred values 

scatter_function(x_increments_array, y_increments_array, 'coral')
plt.title('Wavelength against time (standardised) in index increments of 10')
plt.plot(x_increments_array, y_predicted_k_increments, color='green', label='Order ' + str(k_selected))


# _____________________________________________________________________________________________________________________________________________

# QUESTION 3: BOOTSTRAPPING

# creatting a bootstrapping function that inputs the number of iterations to do bootstrap for, and the actual and predicted data

def bootstrapping(actual_data, predicted_data, iterations):
    bootstrapped_data = []
    residuals_bootstrap = (actual_data - predicted_data).flatten()  # calculating bootstrap residuals array. denoted in notes by \epsilon^{hat}

    for i in range(0, iterations):

        residuals_bootstrap_iteration = np.random.choice(residuals_bootstrap, len(residuals_bootstrap), replace=True)   # getting the bootstrap sample using random choice, WITH replacement. denoted in notes by \epsilon^{*}

        y_response_bootstrap = predicted_data.flatten() + residuals_bootstrap_iteration  # calculated bootstrapped response array using bootstrap_sample and predicted data.  denoted in notes by y^{*}

        y_predicted_bootstrapped = linear_regression(x_increments_array, y_response_bootstrap, k_selected)  # finding the linear regression y_predicted values for the bootstrapped response

        bootstrapped_data.append(y_predicted_bootstrapped)   # storing y_predicted_bootstrapped in a list


    bootstrapped_df = pd.DataFrame(np.stack(bootstrapped_data, axis=1), columns=[str(i) for i in range(iterations)])  # at the end of the for loop, storing the bootsrapped response values in a pandas data frame for all the iterations and 85 x_values

    return bootstrapped_df  # the columns of the df represent the bootstrap iterations, and the rows represent the sampled and standardised time values


# Creating a quantile function to find the 0.025 and 0.975 quantile

def quantile_function(data, quantile_number):    # data is given as a dataframe here
    quantile_array = np.zeros(len(data))
    
    for i in range(0, len(data)):
        quantile_data = data.iloc[i].values   # locates the row from the dataframe to find the quantiles for 
        
        quantile_array[i] = np.quantile(quantile_data, quantile_number)  # adds the quantile for each x_value into an array

    return quantile_array


# calling the functions to perform bootstrapping and getting the quantiles for 

bootstrap_iter = 12  
bootstrapped_df = bootstrapping(y_increments_array, y_predicted_k_increments, bootstrap_iter)

quantile_025 = quantile_function(bootstrapped_df, 0.025)
quantile_975 = quantile_function(bootstrapped_df, 0.975)

# using a plt.fill_between function to plot the 0.025 and 0.975 quantiles to get the 95% pointwise confidence band for the expected wavelength for the regression order of 'k_selected'

plt.fill_between(x_increments_array.squeeze(), quantile_025, quantile_975, alpha = 0.3, label=f'95% Confidence band with {bootstrap_iter} iterations')
plt.legend()


# _____________________________________________________________________________________________________________________________________________

# Question 3(b): Increasing the bootstrapping sample multiple times and graphically showing that the confidence band converges

bootstrap_multiple_iter = [5, 20, 100, 500, 1200, 3000]   # number of times bootstrapping will be performed

x,y = 0,0  # temporary indices used to create a visually appealing subplot

fig,ax = plt.subplots(2,3, dpi=150)
fig.suptitle(f'Subplots showing 95% confidence interval convergence as bootstrap sample increases')

for i in range(0, len(bootstrap_multiple_iter)):

    # performing the bootstrapping and quantile_function functions in a loop for the different bootstrap iterations
    bootstrapped_df = bootstrapping(y_increments_array, y_predicted_k_increments, bootstrap_multiple_iter[i])   

    quantile_025 = quantile_function(bootstrapped_df, 0.025) 
    quantile_975 = quantile_function(bootstrapped_df, 0.975)

    # temp code to create visual subplot for 2x3 plots
    if y == 3:
        y = 0
        x += 1

    # plotting the sampled x,y data; linear regression values and the bootstrapped sample in a subplot
    ax[x, y].plot(x_increments_array, y_increments_array, '.', color='coral')
    ax[x, y].plot(x_increments_array, y_predicted_k_increments, color='green')

    ax[x, y].fill_between(x_increments_array.squeeze(), quantile_025, quantile_975, alpha = 0.3)
    ax[x, y].set_title(f'{bootstrap_multiple_iter[i]} iterations')

    y += 1

for ax in fig.get_axes():
    ax.label_outer()
fig.tight_layout()

# _____________________________________________________________________________________________________________________________________________

plt.show()


