import math
import numpy as np
import pandas as pd
import random
import scipy
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures

random.seed(1847960)

df = pd.read_csv('vrv20.csv')

X = df['X'].values.tolist()
Y = df['Y'].values.tolist()

plt.figure(dpi=150)
plt.plot(X, Y, '.')
plt.title('vrv20: Wavlength (nm) plotted against time')
plt.grid(True)

# _______ import data into 2 lists



plt.show()