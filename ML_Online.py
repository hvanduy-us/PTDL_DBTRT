import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import warnings
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import math

# ve bieu do mo phong
def Plot(X_test,y_test,y_pred):
    # Plot outputs
    plt.scatter(X_test, y_test, color="black")
    plt.plot(X_test, y_pred, color="blue", linewidth=3)

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.xticks(())
    plt.yticks(())

    plt.show()

#thuat giai dong thuan
#input
# models: cac mo hinh hoi quy tuyen tinh cua k(k = 2) chuyen gia
# data: du lieu cho qua trinh huan luyen
# coef: he so cua mo hinh can duoc huan luyen
# T: so luong vong lap cho huan luyen
# epsilon: ngan sach bao mat
# theta: do chinh xac cua mo hinh
#output: 
# regret: sai lam cua cac chuyen gia
# coef: he so moi cua mo hinh     
def RWM(models, data, coef, T = 10, epsilon = 1.0, theta= 0.75):

    eta = epsilon / np.sqrt(32 * T * np.log(1 / theta)) 
    k = len(models)
    n = len(data[0])
    l_t = np.zeros(k)
    
    #khoi tao trong so = 1 cho k chuyen gia
    Ws = np.ones(k)
    for t in range(T):
        weights_for_choices = [0.5, 0.5]

        choices = np.array(random.choices(list(enumerate(models)), weights = weights_for_choices, k = n))
        print("coef t = ", t, " = ", coef )

        for j in range(n):
            # Chuyen gia duoc chon
            i = choices[j][0]

            y_pred_model = choices[j][1].predict([data[0][j]])
            # l_i_t = | y_pred_model - y |  theo mo hinh S1 | S2
            l_i_t = y_pred_model - data[1][j]

            # Ax + b = y
            # A' = ((y + dental) - b) / (x + dental)
            coef = (y_pred_model - choices[j][1].intercept_) / (data[0][j] + l_i_t)

            #cap nhap Ws
            Ws[i] *= np.exp((-eta) * l_i_t)

            # Tong mat mat
            l_t[i] += l_i_t
    #trung binh mat mat cua chuyen gia i tren vong  T
    l_t /= T 

    #Sai lam cua cac chuyen gia 
    l_t -= np.amin(l_t)
    return l_t, coef
    
def main():
    #doc du lieu tu file weight_height vao data
    data = pd.read_csv('weight-height.csv')
    data.head()

    #chia du lieu thanh 3 phan D1, D2, D3
    D_size = int(len(data) * 0.4)
    D1 = data[:D_size]
    D2 = data[D_size:D_size*2]
    D3 = data[: int(D_size / 2)]

    # Chia du lieu cho phan train va test
    X1 = D1.copy().drop(["Gender" ,"Weight"],axis = 1)
    y1 = D1["Weight"]
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

    X2 = D2.copy().drop(["Gender" ,"Weight"],axis = 1)
    y2 = D2["Weight"]
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

    X3 = D3.copy().drop(["Gender" ,"Weight"],axis = 1)
    y3 = D3["Weight"]
    X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X3, y3, test_size=0.2, random_state=42)

    #huan luyen cho 2 du lieu voi 3 mo hinh S, S1, S2
    S1 = linear_model.LinearRegression()
    S1.fit(X_train_1, y_train_1)
    y_pred_1 = S1.predict(X_test_1)
    MSE_1 = mean_squared_error(y_test_1, y_pred_1, squared=False)

    S2 = linear_model.LinearRegression()
    S2.fit(X_train_2, y_train_2)
    y_pred_2 = S2.predict(X_test_2)
    MSE_2 = mean_squared_error(y_test_1, y_pred_2, squared=False)

    S = linear_model.LinearRegression()
    S.fit(X_train_3, y_train_3)
    y_pred = S.predict(X_test_3)
    MSE = mean_squared_error(y_test_3, y_pred, squared=False)

    warnings.filterwarnings("ignore")
    # Chuẩn bị dữ liệu để chạy thuật giải
    experts = [S1, S2]
    train_data = []
    train_data.append(np.array(X_train_3))
    train_data.append(np.array(y_train_3))

    Regret, coef = RWM(experts, train_data, S.coef_, 20)

    S.coef_ = coef

    # Chạy thử mô hình
    y_pred_model = S.predict(X_test_3)
    MSE = mean_squared_error(y_test_3, y_pred_model, squared=False)
    # Regret
    print("Mat mat",Regret)
    # The coefficients
    print("Coefficients: \n", S.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test_3, y_pred_model))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test_3, y_pred_model))
    Plot(X_test_3,y_test_3,y_pred_model)


if __name__ == "__main__":
    main()