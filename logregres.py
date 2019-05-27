#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:10:49 2019

@author: xiangshuyang
"""
import numpy as np

class Logregre:
    
    def __init__(self,DataMat,ClassLabel,alpha):
        self.DataMat=DataMat
        self.ClassLabel=ClassLabel
        self.DataMatrix= np.mat(DataMat)
        self.LabelMatrix=np.mat(self.ClassLabel).transpose()
        self.alpha=alpha

    def sigmoid(self,inX): #sigmoid function 
        return 1.0/(1+np.exp(-inX)) 


    def gradAscent(self):
        m,n = np.shape(self.DataMatrix)
        weights = np.ones((n,1))
        weights0 = np.ones((n,1))
        beta=1
        for i in range(500):
            if beta<10^(-3):
                break
            else:
                weights0=weights
                h = self.sigmoid(self.DataMatrix*weights)  
                error = h-self.LabelMatrix
                weights = weights -self.alpha * self.DataMatrix.transpose()* error
                beta=np.sum(np.abs(weights0-weights))
        return weights

    def plotBestFit(self):
        import matplotlib.pyplot as plt
        wei=self.gradAscent()
        weights = np.asarray(wei)
        dataArr = np.array(self.DataMat)
        n = np.shape(dataArr)[0]
        xcord1 = []
        ycord1 = []
        xcord2 = []
        ycord2 = []
        for i in range(n):
            if self.ClassLabel[i]==1:
               xcord1.append(dataArr[i,1])
               ycord1.append(dataArr[i,2])
            else:
                xcord2.append(dataArr[i,1])
                ycord2.append(dataArr[i,2])
        plt.scatter(xcord1, ycord1, s=40, c='red', alpha=0.5)
        plt.scatter(xcord2, ycord2, s=40, c='blue', alpha=0.5)
        x = np.arange(-3.0, 3.0, 0.1)
        y = (-weights[0]-weights[1]*x)/weights[2]
        plt.plot(x, y,c='black')
        

######Test###########
alpha=0.001
dataMat = []; labelMat = []
fr = open('dataset.txt')
for line in fr.readlines():
    lineArr = line.strip().split()
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    labelMat.append(int(lineArr[2]))
g=Logregre(dataMat,labelMat,alpha)
g.plotBestFit()