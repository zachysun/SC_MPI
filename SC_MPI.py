#!/usr/bin/env python
# coding: utf-8

get_ipython().system(' pip install sklearn')

import numpy as np
from keras.datasets import mnist
from numpy import linalg as la
from sklearn.cluster import AgglomerativeClustering

def get_mnist():
    np.random.seed(123)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_all = np.concatenate((x_train, x_test), axis = 0)
    Y = np.concatenate((y_train, y_test), axis = 0)
    X = x_all.reshape(-1,x_all.shape[1]*x_all.shape[2])
    
    p = np.random.permutation(X.shape[0])
    X = X[p].astype(np.float32)*0.02
    Y = Y[p]
    return X[:10000], Y[:10000]

X, Y  = get_mnist()

n=len(Y)
K = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        K[i,j] = np.dot(X[i],X[j])

# Create G
def convert(arr): 
    
    x_num = arr.shape[0]
    y_num = arr.shape[1]
    
    row = []
    line = []
    
    g = np.zeros((x_num,x_num))
    
    ## constraint_a
    for i in range(y_num):
        for j in  range(x_num):
            for k in range(x_num):
                if (arr[j,i] == arr[k,i] and arr[j,i] == 1) or j == k:
                    g[j,k] = 1 
                    g[k,j] = 1 
    
    ## constraint_b
    for i in range(y_num):
        for j in range(x_num):
            if arr[j,i] == 1:   
                row.append(j)
                line.append(i)
                
    num = len(row)
    
    for m in range(num):
        for n in range(num):
            if  line[m] != line[n]:
                g[row[m],row[n]] = -1
                g[row[n],row[m]] = -1
          
    ## constraint_c
    for i in range(y_num):
        for j in  range(x_num):
            for k in range(x_num):
                if (arr[j,i] == 1 and arr[k,i] == -1) or (arr[j,i] == -1 and arr[k,i] == 1):
                    g[j,k] = -1 
                    g[k,j] = -1 
    
    ## constraint_d and constraint_e
    km = []
    
    for i in range(x_num):
        if arr[i].sum() == -(y_num-1):
            km.append(i)
    
    km_num = len(km)
    
    for i in range(km_num):
        for j in range(km_num):
            if (arr[km[i]]*arr[km[j]]).sum() == y_num-1:
                g[km[i],km[j]] = 1
                g[km[j],km[i]] = 1
            else:
                g[km[i],km[j]] = -1
                g[km[j],km[i]] = -1
           
    return g

def convert_label_constraints(Y, k):
    num = len(Y)
    a = np.zeros((num,k))
    for i in range(num):
        a[i,Y[i]] = 1
    b = convert(a)
    return b

G=convert_label_constraints(Y,10)

# k & alpha
k = 10
alpha = 1

# Hierarchical clustering

# Step1 and Step2
def get_k_eig_ve(K,k,alpha,*args):
    n = len(K)
    GG = np.zeros((n,n))
    for arg in args:
        GG = GG + arg
    KG = K + alpha*GG
    KG = KG.reshape(n,n)
    eig_va, eig_ve = la.eig(KG)
    return eig_ve[:k]


u = get_k_eig_ve(K,10,1,G)
u_t = u.T

clustering = AgglomerativeClustering(n_clusters=10,linkage='ward').fit(u_t)

U = clustering.labels_
U

# accuracy
def accuracy(Y,U):
    right=0
    for i in range(len(U)):
        if Y[i] == U[i]:
            right = right+1
    acc = right/len(U)
    return acc

accuracy(Y,U)

clustering = AgglomerativeClustering(n_clusters=10,linkage='ward').fit(X)
UX = clustering.labels_
accuracy(Y,UX)

clustering = AgglomerativeClustering(n_clusters=10,linkage='single').fit(X)
UXX = clustering.labels_
accuracy(Y,UXX)

clustering = AgglomerativeClustering(n_clusters=10,linkage='complete').fit(X)
UXXX = clustering.labels_
accuracy(Y,UXXX)

clustering = AgglomerativeClustering(n_clusters=10,linkage='average').fit(X)
UXXXX = clustering.labels_
accuracy(Y,UXXXX)

from sklearn.cluster import KMeans
clf = KMeans(n_clusters=10,init='k-means++')
S = clf.fit(X)
UXXXXX=clf.labels_
accuracy(Y,UXXXX)