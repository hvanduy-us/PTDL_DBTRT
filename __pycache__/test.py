import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from array  import *
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import math


    # Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
diabetes_X_train_1 = diabetes_X[0:200]
diabetes_X_train_2 = diabetes_X[200:422]

diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train_1 = diabetes_y[0:200]
diabetes_y_train_2 = diabetes_y[200:422]
diabetes_y_test = diabetes_y[-20:] 
print(diabetes_X_train_1)