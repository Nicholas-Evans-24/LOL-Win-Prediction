# -*- coding: utf-8 -*-
"""

Dataset: https://www.kaggle.com/bobbyscience/league-of-legends-diamond-ranked-games-10-min

@author: Nicholas Evans
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import decomposition,preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier,LocalOutlierFactor
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt

def read():
    
    df = pd.read_csv(Path("high_diamond_ranked_10min.csv"))
    return df
    
def train(df):
    
    
    #Display Correlation of current DF
    correlation_mat = df.corr().abs()
    plt.figure(figsize=(30,30))
    sns.heatmap(correlation_mat,annot=True,linewidths=1)
    plt.show()
    
    upper = correlation_mat.where(np.triu(np.ones(correlation_mat.shape),k=1).astype(bool))
    
    #Drops Columns that have correlation higher than 0.9
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    df = df.drop(to_drop,axis=1)
   
    #Display Correlation of current DF
    correlation_mat = df.corr().abs()
    plt.figure(figsize=(30,30))
    sns.heatmap(correlation_mat,annot=True,linewidths=1)
    plt.show()
    
    
    #Drop columns with correlation to BlueWins less than 0.1
    df = df.drop(["blueWardsPlaced","blueWardsDestroyed","redWardsPlaced","redWardsDestroyed","blueHeralds","redHeralds","gameId"],axis=1)
    
    #Display Correlation of current DF
    correlation_mat = df.corr().abs()
    plt.figure(figsize=(30,30))
    sns.heatmap(correlation_mat,annot=True,linewidths=1)
    plt.show()
    
    #determines outliers in each column and eliminates rows
    lof = LocalOutlierFactor(n_neighbors=40)
    middle = lof.fit_predict(df)
    
    rows, cols = df[middle == -1].shape
    print("Number of outliers: ", rows)
    
    df = df[middle != -1]
    
    
    
    # Create target and remove gameId
    y = df["blueWins"]
    x = df.drop(["blueWins"],axis=1)
    
    #normalize the training data
    names = x.columns
    x = preprocessing.normalize(x)
    x = pd.DataFrame(x,columns=names)
    
    #Renamed target values
    y = y.replace(0,"Lose")
    y = y.replace(1,"Win")

   
    #split data to train and test
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
    
    #Shortens columns for faster learning
    pca = decomposition.PCA(n_components=2)
    pca.fit(x_train)
    x_model = pca.transform(x_train)
    x_model = pd.DataFrame(x_model)
    
    #Shows us how much data is saved 
    #sum(list) / 1 = % data saved
    print("explained_variance_ratio: \n",pca.explained_variance_ratio_)
   
    #Logistic Regression Model
    lg = LogisticRegression()
    lg.fit(x_model,y_train)
    x_test_model= pca.fit_transform(x_test)
    
    print()
    print("log-Reg Score: ",lg.score(x_test_model,y_test))
    
    #KNeighbors Model
    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(x_model,y_train)
    print("k-neighbors Score: ",knn.score(x_test_model,y_test))
    
    #Support Vector Classification Model 
    svc = SVC(gamma='auto')
    svc.fit(x_model,y_train)
    print("SVC Score: ",svc.score(x_test_model,y_test))
    
    
df = read()
train(df)