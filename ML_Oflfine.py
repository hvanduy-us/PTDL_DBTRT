import numpy as np
import math
import random

#Tung dong xu
# Lan 1: ngua thi tra loi dung hien trang
#        Sap thi tung lan 2
#Lan 2: Ngua thi tra loi 1("Yes")
#       Sap tra loi 0("No")
def Toss_a_Coin(data):
    result = np.empty((0),dtype=int)

    for i in range(len(data)): 
        if(random.randint(0, 1) == 0):
            result = np.append(result,data[i])
        else:
            if(random.randint(0, 1) == 0):
                result = np.append(result,1)
            else:
                result = np.append(result,0)
        
    return result

# Ham Uoc luong xac xuat
def UocLuong_XacSuat( p, q):
    Pr_Yes = (2*p*q+1-q)/2
    Pr_Yes_P = (q + 1)/2 
    return Pr_Yes, Pr_Yes_P
 
# Ham sai so xac suat
def SaiSo_XacSuat(p, q):
    Pr_P_Yes = (p*(1+q))/(1-q+2*pq)
    Pr_P_No = (p*(1-q))/(1+q-2*p*q)
    OR1 = (1+q)/(1-q+2*p*q)
    OR2 = ((1+q)*(1+q-2*p*q))/((1-q)*(1-q+2*p*q))

    return Pr_P_Yes, Pr_P_No, OR1, OR2

# Xac dinh do bao mat cua thua toan voi co mau va do chinh xac
def Do_Bao_Mat(n,SE):
    q = math.sqrt(1/(n * SE * SE))    
    #SE = 1 / q

    if( q > 1):
        print('Xac xuat vuot nguong!')
        q = 1
    return q

# tinh co mau uoc luong
def Co_Mau(q,SE):
    n = 1/(q * q * SE * SE)

    return q

def main():
    ##x = np.array(['Ali','Join', 'Bob','Jack','Donal']).reshape((-1, 1))
    y = np.random.randint(0,2,500).reshape((-1, 1))
    result = Toss_a_Coin(y)
    q = Do_Bao_Mat(len(result), 0.05)
    print('ket qua thong ke: ', result)
    print('Do chinh xac Detal: ',0.05)
    print('Do bao mat voi co mau ',len(result),': ',q)
    #print('Sai so chuan theo q: ', 1 / q)
if __name__ == "__main__":
    main()
        
    