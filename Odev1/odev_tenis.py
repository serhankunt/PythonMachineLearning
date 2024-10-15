# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:39:42 2024

@author: hserh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('odev_tenis.csv')

from sklearn import preprocessing
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)

c=veriler2.iloc[:,:1]

from sklearn import preprocessing
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)

havadurumu = pd.DataFrame(data=c,index = range(56) , columns=['o','r','s'])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis=1)
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler],axis=1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

import statsmodels.api as sm

# X= np.append(arr=np.ones((14,1)).astype(int),values=sonveriler.iloc[:,:-1],axis=1)

# X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
# X_l = np.array(X_l,dtype=float)
# model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
# print(model.summary())

# 1. Verilerin hazırlanması
X = np.append(arr=np.ones((56, 1)).astype(int), values=sonveriler.iloc[:, :-1], axis=1)  # Sabit terim ekleme
y = sonveriler.iloc[:, -1].values  # Bağımlı değişken

# 2. Backward Elimination fonksiyonunu tanımlama
def backward_elimination(X, y, sl=0.4):
    num_vars = X.shape[1]
    for i in range(num_vars):
        model = sm.OLS(y, X).fit()
        max_p_value = max(model.pvalues).astype(float)  # En yüksek p-değeri
        if max_p_value > sl:  # p > 0.4 ise
            for j in range(num_vars - i):
                if model.pvalues[j].astype(float) == max_p_value:
                    X = np.delete(X, j, 1)  # O kolonu sil
                    break
        
        else:
            break
    print(model.summary())
    return X

# 3. Backward Elimination işlemi
SL = 0.15  # Anlamlılık seviyesi
X_opt = sonveriler.iloc[:, :-1].values  # Bağımsız değişkenler
X_opt = np.append(arr=np.ones((56, 1)).astype(int), values=X_opt, axis=1)  # Sabit terim ekleme

X_Modeled = backward_elimination(X_opt, y, SL)

    
    

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)





# import statsmodels.api as sm

# X= np.append(arr=np.ones((14,1)).astype(int),values=sonveriler.iloc[:,:-1],axis=1)

# sol = sonveriler.iloc[:,:3]
# deger = sonveriler.iloc[:,3]
# sag = sonveriler.iloc[:,4:]

# # sol_df = pd.DataFrame(sol)
# # sag_df = pd.DataFrame(sag)

# X_l = pd.concat([sol,sag],axis=1)
# X_l = np.array(X_l,dtype=float)

# model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
# print(model.summary())

# play = veriler.iloc[:,-1:].values
# print(play)

# from sklearn import preprocessing

# le = preprocessing.LabelEncoder()

# play[:,-1] = le.fit_transform(veriler.iloc[:,-1])
# print(play) 

# windy = veriler.iloc[:,-2:-1].values
# print(windy)

# from sklearn import preprocessing

# le = preprocessing.LabelEncoder()

# windy[:,-1] = le.fit_transform(veriler.iloc[:,-1])
# print(windy)



# print(veriler)