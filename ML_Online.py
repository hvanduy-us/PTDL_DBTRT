import matplotlib.pyplot as plt
import numpy as np
import random
from array  import *
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#Create Data
def dataset():      
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

    return diabetes_X_train_1,diabetes_X_train_2,diabetes_X_test, diabetes_y_train_1,diabetes_y_train_2,diabetes_y_test

#Linear Regresion
def LinearRegression(X_train, y_train):
    #Create linear regression object
    regr = linear_model.LinearRegression()

    #Train the model using the training sets
    regr.fit(X_train, y_train)

    return regr

def predict(regr, X_test):
    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(X_test)
    return diabetes_y_pred

def Plot(X_test,y_test,y_pred):
    # Plot outputs
    plt.scatter(X_test, y_test, color="black")
    plt.plot(X_test, y_pred, color="blue", linewidth=3)

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.xticks(())
    plt.yticks(())

    plt.show()

def Loss_Func(yt,y):
    loss_f=[]
    for i in range(len(y)):
        loss_f.append(abs(yt[i] - y[i]))
    return loss_f
def main():
    diabetes_X_train_1,diabetes_X_train_2,diabetes_X_test, diabetes_y_train_1, diabetes_y_train_2,diabetes_y_test = dataset()

    regr1 = LinearRegression(diabetes_X_train_1, diabetes_y_train_1)
    regr2 = LinearRegression(diabetes_X_train_2, diabetes_y_train_2)

    X_test_1 = diabetes_X_test[0:10]
    X_test_2 = diabetes_X_test[10:20]

    diabetes_y_pred_1 = predict(regr1, X_test_1)
    diabetes_y_pred_2 = predict(regr2, X_test_2)

    Y_pred_train = np.concatenate((diabetes_y_pred_1, diabetes_y_pred_1), axis=0)

    regr = LinearRegression(diabetes_X_test, Y_pred_train)
    y_pred = predict(regr, diabetes_X_test)
    # F= AX+b
    #The coefficients
    print("Coefficients(b): \n", regr.coef_)
     #The coefficients
    print("Intercrept(A): \n", regr.intercept_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, y_pred))

    Plot(diabetes_X_test, Y_pred_train, y_pred)


if __name__ == "__main__":
    main()