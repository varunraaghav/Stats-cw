# HOW TO DO SUBPLOTS (IE PLOTS SIDE BY SIDE) 

# fig,ax = plt.subplots(ncols=2)

# ax[0].plot(X, Y, '.')
# ax[0].set_title('blue dots')


# ax[1].plot(X, Y, 'x', color='red')
# ax[1].set_title('red x')

# ______________________________________________________________________________________________

import numpy as np
import matplotlib.pyplot as plt

# create some sample data
x = np.array([1, 2, 3, 4, 5])
y_actual = np.array([2.2, 4.5, 6.3, 8.1, 10.2])
y_predicted = np.array([1.8, 4.0, 6.2, 8.4, 10.6])

# plot the actual values as scatter plot
plt.scatter(x, y_actual, color='blue', label='Actual')

# plot the predicted values as a line plot
plt.plot(x, y_predicted, color='red', label='Predicted')

# plot a line connecting the actual and predicted values
for i in range(len(x)):
    plt.plot([x[i], x[i]], [y_actual[i], y_predicted[i]], color='gray', linestyle='--')

# set plot title and labels
plt.title('Actual vs Predicted Values')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# display the plot
plt.show()
