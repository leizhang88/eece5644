# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:46:24 2020

@author: lei
"""

import numpy as np
from scipy import linalg, random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

random.seed(2345)

# sample dimension
n = 3

# sample size
N = 10000

# class prior
class_prior = np.array([.2, .25, .25, .3])
NC = len(class_prior)

# class conditional pdf: gaussian
m = np.array([[0, 0, 0],
              [1, 0, 0],
              [1, 0, 1],
              [0, 0, 1]]).T

# eigenvalues gaussian covariance matrices
lam = np.array([.12, .11, .17, .14])

# sampling and labeling
X = np.zeros((n, N))
rnd = random.rand(N)

temp = np.insert(np.cumsum(class_prior), 0, 0)

y = np.zeros(N)
for i in range(NC):
    ind = (rnd >=temp[i]) & (rnd < temp[i+1])
    y[ind] = i + 1
    X[:, ind] = random.multivariate_normal(m[:, i], np.eye(n) * lam[i],
     np.sum(ind)).T



#-------------------------------------------------------------------------
# Part A: minimum probability of error classification (MAP classification)
#-------------------------------------------------------------------------
decisionMAP = np.zeros(N)

for i in range(N):
    conditional = np.zeros(NC)
    
    for j in range(NC):
        conditional[j] = multivariate_normal.pdf(X[:, i], mean=m[:, j], 
                   cov=(np.eye(n) * lam[j]))
    
    decisionMAP[i] = np.argmax(conditional * class_prior) + 1

# confision matrix
CM = np.zeros((NC, NC))

for i in range(NC):
    for j in range(NC):
        CM[i, j] = ((decisionMAP == i+1) & (y == j+1)).sum() / (y == j+1).sum()

markers = ['.', 'o', '^', 's']

# plot samples and decisions
fig1 = plt.figure(figsize=(4, 4), dpi=150)
ax1 = fig1.gca()
for i in range(NC):
    ax1.scatter(X[0, ((decisionMAP==i+1) & (y==i+1))], 
              X[2, ((decisionMAP==i+1) & (y==i+1))], 
              marker=markers[i], c='green', alpha=.1)
    ax1.scatter(X[0, ((~(decisionMAP==i+1)) & (y==i+1))], 
              X[2, ((~(decisionMAP==i+1)) & (y==i+1))], 
              marker=markers[i], c='red', alpha=.1)
ax1.set_xlim((-2, 3))
ax1.set_ylim((-2, 3))
ax1.set_title('MAP classifer')
plt.annotate(r'$L_1$', xy=(-1.5, -1.5))
plt.annotate(r'$L_2$', xy=(2.5, -1.5))
plt.annotate(r'$L_3$', xy=(2.5, 2.5))
plt.annotate(r'$L_4$', xy=(-1.5, 2.5))
plt.tight_layout()
plt.savefig('Q2A.pdf')
plt.show()

#----------------------------------------------
# Part B: ERM classifier with given loss matrix
#----------------------------------------------
# loss matrix
Loss = np.array([[0, 1, 2, 3],
                 [10, 0, 5, 10],
                 [20, 10, 0, 1],
                 [30, 20, 1, 0]])

decisionERM = np.zeros(N)

for i in range(N):
    posterior = np.zeros((NC, 1))
    
    for j in range(NC):
        posterior[j] = multivariate_normal.pdf(X[:, i], 
                 mean=m[:, j], cov=(np.eye(n) * lam[j])) * class_prior[j]
    
    decisionERM[i] = np.argmin(Loss.dot(posterior)) + 1

# plot samples and decisions
fig2 = plt.figure(figsize=(4, 4), dpi=150)
ax2 = fig2.gca()
for i in range(NC):
    ax2.scatter(X[0, ((decisionERM==i+1) & (y==i+1))], 
              X[2, ((decisionERM==i+1) & (y==i+1))], 
              marker=markers[i], c='green', alpha=.1)
    ax2.scatter(X[0, ((~(decisionERM==i+1)) & (y==i+1))], 
              X[2, ((~(decisionERM==i+1)) & (y==i+1))], 
              marker=markers[i], c='red', alpha=.1)
ax2.set_xlim((-2, 3))
ax2.set_ylim((-2, 3))
ax2.set_title(r'ERM classifer with given loss')
plt.annotate(r'$L_1$', xy=(-1.5, -1.5))
plt.annotate(r'$L_2$', xy=(2.5, -1.5))
plt.annotate(r'$L_3$', xy=(2.5, 2.5))
plt.annotate(r'$L_4$', xy=(-1.5, 2.5))
plt.tight_layout()
plt.savefig('Q2B.pdf')
plt.show()
                
# expected loss
expect_loss = Loss[(decisionERM-1).astype(int), (y-1).astype(int)].sum() / N
