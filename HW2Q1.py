# -*- coding: utf-8 -*-
"""
EECE5644 Intro2ML - homework2, question1

Approximate class posterior with logistic linear/quadratic function
whose parameters are estimated with MLE
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import random
from sklearn import mixture
from scipy.stats import multivariate_normal
from scipy.optimize import minimize


def generateCov(d, cov_value):
    M = cov_value.size
    Covs = np.zeros((M, d, d))
    
    for i in range(M):
        Covs[i, :, :] = np.eye(d) * cov_value[i]
    
    return Covs

# generate GMM samples
def generateGMM(d, N, alpha, means, Covs):
    alpha_cum = np.insert(np.cumsum(alpha), 0, 0)

    X = np.zeros((N, d))
    temp = random.rand(N)
    gmm_label = np.zeros(N)

    for i in range(alpha.size):
        ind = (temp >= alpha_cum[i]) & (temp < alpha_cum[i + 1])
        gmm_label[ind] = i + 1
        X[ind, :] = random.multivariate_normal(means[i, :].flatten(), 
         Covs[i, :, :], ind.sum())
    
    return X, gmm_label


def evaluateGMM(X, alpha, means, Covs):
    '''evaluate GMM pdf at each row of X'''
    
    N = X.shape[0]
    gmm_pdf = np.zeros(N)
    
    for i in range(alpha.size):
        rv = multivariate_normal(mean=means[i, :].ravel(), 
                                 cov=Covs[i, :, :])
        gmm_pdf = gmm_pdf + alpha[i] * rv.pdf(X)
    
    return gmm_pdf


def generateSamples(N, means0, Covs0, m1, C1, class_prior):
    '''generate samples, shape(N, d)'''
    # sample dimension
    d = m1.size
    
    # preallocation
    N = np.int(N)
    X = np.zeros((N, d))
    y = np.zeros(N)
    
    temp = random.rand(N)
    
    # class 0: GMM
    ind0 = (temp <= class_prior[0])
    y[ind0] = 0
    
    # components number
    M = Covs0.shape[0]
    
    weights = np.ones(M) / M
    
    X[ind0, :], dummy = generateGMM(d, ind0.sum(), weights, means0, Covs0)
    
    # class 1: Gaussian
    ind1 = (temp > class_prior[0])
    y[ind1] = 1
    
    X[ind1, :] = random.multivariate_normal(m1, C1, ind1.sum())
    
    return X, y

#----------------------------------------------------------
# Part 1: generate tarining datasets and validation dataset
# and classify with true class pdf
#----------------------------------------------------------
random.seed(1234)

# class prior
class_prior = np.array([.6, .4])

# class pdf
means0 = np.array([[5, 0], [0, 4]])
Covs0 = np.array([[[4, 0], [0, 2]], [[1, 0], [0, 3]]])
weights = np.array([.5, .5])

m1 = np.array([3, 2])
C1 = np.array([[2, 0], [0, 2]])

# training dataset of 100
X1, y1 = generateSamples(1e2, means0, Covs0, m1, C1, class_prior)

# training dataset of 1000
X2, y2 = generateSamples(1e3, means0, Covs0, m1, C1, class_prior)

# training dataset of 10000
X3, y3 = generateSamples(1e4, means0, Covs0, m1, C1, class_prior)

# validation dataset of 20000
X4, y4 = generateSamples(2e4, means0, Covs0, m1, C1, class_prior)


#----------------------------------
# Part 1: training on X4 using ERM
#----------------------------------

def ERMclassifer(X, y, class_prior, weights, means0, Covs0, m1, C1, 
                 dataset_name, figname):
    '''calculate discriminant score and draw decision boundary'''
    
    # penalty matrix
    penalty = np.ones((2, 2)) - np.eye(2)
    
    # class_conditional
    class_conditional = np.zeros((X.shape[0], class_prior.size))
    
    class_conditional[:, 0] = evaluateGMM(X, weights, means0, Covs0)
    
    class_conditional[:, 1] = multivariate_normal(m1, C1).pdf(X)
    
    discriminant_score = class_conditional[:, 1] / class_conditional[:, 0]
    
    # class posterior
    class_posterior = class_conditional.dot(np.diag(class_prior))
    class_posterior = class_posterior / \
        np.sum(class_posterior, axis=1).reshape(-1, 1).dot(np.ones((1, 2)))
    expected_risk = class_posterior.dot(penalty)
    
    decision = np.argmin(expected_risk, axis=1)
    
    ind00 = (y == 0) & (decision == 0)
    ind01 = (y == 1) & (decision == 0)
    ind10 = (y == 0) & (decision == 1)
    ind11 = (y == 1) & (decision == 1)
    
    plt.figure(figsize=[3, 3], dpi=150)
    plt.scatter(X[ind00, 0], X[ind00, 1], s=7, marker='o', color='g', alpha=.3, 
                facecolors='none', label='L0: correct')
    plt.scatter(X[ind01, 0], X[ind01, 1], marker='+', color='r', alpha=.3,
                label='L1: wrong')
    plt.scatter(X[ind10, 0], X[ind10, 1], s=7, marker='o', color='r', alpha=.3, 
                facecolors='none', label='L0: wrong')
    plt.scatter(X[ind11, 0], X[ind11, 1], marker='+', color='g', alpha=.3,
                label='L1: correct')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Training based on ' + 
              r'$D_{{{}}}^{{{}}}$'.format(dataset_name[0], dataset_name[1]))
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.axis('equal')
    plt.savefig('../hw2/decision_' + figname + '.pdf')
    plt.show()
    
    return discriminant_score

def ROCcurve(discriminant_score, sample_label, class_prior, 
             dataset_name, figname):
    '''find the theoretically minimum probability of error 
    and plot ROC curve'''
    
    log_score = np.log(discriminant_score)
    # sort discriminant score in ascending order
    sorted_score = np.sort(log_score)
    
    # objective value
    eps = 1e-8
    gamma = (sorted_score[0:-1] + sorted_score[1:]) / 2
    gamma = np.insert(gamma, 0, sorted_score[0] - eps)
    gamma = np.append(gamma, sorted_score[-1] + eps)
    
    p10 = np.zeros(gamma.size)
    p11 = np.zeros(gamma.size)
    p01 = np.zeros(gamma.size)
    p_error = np.zeros(gamma.size)
    
    for i, value in enumerate(gamma):
        dcision = log_score > value
        
        # positive for class 1
        p10[i] = (dcision & (sample_label == 0)).sum() / (sample_label == 0).sum()
        p11[i] = (dcision & (sample_label == 1)).sum() / (sample_label == 1).sum()
        p01[i] = (~dcision & (sample_label == 1)).sum() / (sample_label == 1).sum()
        
        p_error[i] = p10[i] * class_prior[0] + p01[i] * class_prior[1]
        
    # minimum probability of error
    min_ind = np.argmin(p_error)
    
    # plot ROC curve and show the point where 
    # the probability of error is minimum
    f1 = plt.figure(figsize=[3, 3], dpi=150)
    ax1 = f1.gca()
    ax1.plot(p10, p11, linewidth=1)
    ax1.scatter(p10[min_ind], p11[min_ind], marker='x', color='red', 
                label=r'min $P_{error}$')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xlabel(r'$P_{FP}$')
    ax1.set_ylabel(r'$P_{TP}$')
    ax1.set_title('Training based on ' + 
                  r'$D_{{{}}}^{{{}}}$'.format(dataset_name[0], dataset_name[1]))
    ax1.legend()
    plt.tight_layout()
    plt.savefig('../hw2/roc_' + figname + '.pdf')
    plt.show()
    
    f2 = plt.figure(figsize=[3, 3], dpi=150)
    ax2 = f2.gca()
    ax2.scatter(gamma[min_ind], p_error[min_ind], s=5, color='red')
    ax2.plot(gamma, p_error, linewidth=1)
    plt.annotate(f'({gamma[min_ind]:.3},{p_error[min_ind]:.3})', fontsize=6, 
                 xy=(gamma[min_ind] - .5, p_error[min_ind]), 
                 xytext=(gamma[min_ind] - 12, p_error[min_ind]), 
                 arrowprops=dict(arrowstyle='->'))
    ax2.set_xlabel(r'$\gamma$')
    ax2.set_ylabel(r'$P_{error}$')
    ax2.set_title('Training based on ' + 
                  '$D_{{{}}}^{{{}}}$'.format(dataset_name[0], dataset_name[1]))
    plt.tight_layout()
    plt.savefig('../hw2/perror_' + figname + '.pdf')
    plt.show()
    
    return


discriminant_score = ERMclassifer(X4, y4, class_prior, weights, means0, Covs0, 
                                  m1, C1, ['validate', '20k'], 'set4')


ROCcurve(discriminant_score, y4, class_prior, ['validate', '20k'], 'set4')


#-------------------------------------------
# Part2.a: ERM classification on 10k dataset
#-------------------------------------------

# estimate class priors based on occurances
class_prior_3 = np.array([(y3 == 0).sum(), (y3 == 1).sum()]) / y3.size

# estimate class-0 pdf using EM
gmm3 = mixture.GaussianMixture(n_components=2, covariance_type='full', 
                                tol=1e-3, reg_covar=1e-6, max_iter=500, 
                                n_init=5).fit(X3[y3==0, :])

# clss-1 pdf
m1_3 = np.mean(X3[y3==1, :], axis=0)

C1_3 = X3[y3==1, :].T.dot(X3[y3==1, :]) / (y3 == 1).sum()

discriminant_score_3 = ERMclassifer(X4, y4, class_prior_3, gmm3.weights_, 
                                    gmm3.means_, gmm3.covariances_, 
                                  m1_3, C1_3, ['train', '10k'], 'set3')

ROCcurve(discriminant_score_3, y4, class_prior_3, ['train', '10k'], 'set3')


#--------------------------------------------
# Part2.b: ERM classification on 1000 dataset
#--------------------------------------------

# estimate class priors based on occurances
class_prior_2 = np.array([(y2 == 0).sum(), (y2 == 1).sum()]) / y2.size

# estimate class-0 pdf using EM
gmm2 = mixture.GaussianMixture(n_components=2, covariance_type='full', 
                                tol=1e-3, reg_covar=1e-6, max_iter=500, 
                                n_init=5).fit(X2[y2==0, :])

# clss-1 pdf
m1_2 = np.mean(X2[y2==1, :], axis=0)

C1_2 = X2[y2==1, :].T.dot(X2[y2==1, :]) / (y2 == 1).sum()

discriminant_score_2 = ERMclassifer(X4, y4, class_prior_2, gmm2.weights_, 
                                    gmm2.means_, gmm2.covariances_, 
                                  m1_2, C1_2, ['train', '1000'], 'set2')

ROCcurve(discriminant_score_2, y4, class_prior_2, ['train', '1000'], 'set2')


#-------------------------------------------
# Part2.c: ERM classification on 100 dataset
#-------------------------------------------

# estimate class priors based on occurances
class_prior_1 = np.array([(y1 == 0).sum(), (y1 == 1).sum()]) / y1.size

# estimate class-0 pdf using EM
gmm1 = mixture.GaussianMixture(n_components=2, covariance_type='full', 
                                tol=1e-3, reg_covar=1e-6, max_iter=500, 
                                n_init=5).fit(X1[y1==0, :])

# clss-1 pdf
m1_1 = np.mean(X1[y1==1, :], axis=0)

C1_1 = X1[y1==1, :].T.dot(X1[y1==1, :]) / (y1 == 1).sum()

discriminant_score_1 = ERMclassifer(X4, y4, class_prior_1, gmm1.weights_, 
                                    gmm1.means_, gmm1.covariances_, 
                                  m1_1, C1_1, ['train', '100'], 'set1')

ROCcurve(discriminant_score_1, y4, class_prior_1, ['train', '100'], 'set1')


#---------------------------------------------------------------------
# Part3.a: approximate class posterior with a logistic-linear-function
#---------------------------------------------------------------------

# def approx_posterior_classifier1(w, X, y, dataset_name):
#     '''classification based on approximate class posterior, 
#     which is evaluated by a logistic-linear-function'''
    
#     # logistic-linear-function
#     h = lambda x: 1 / (1 + np.exp(-x))
    
#     Z = np.block([[np.ones((1, y.size))], [X.T]]).T
    
#     approx_posterior = np.zeros((X.shape[0], 2))
    
#     # class-1: P(L=1) = h(w*x)
#     approx_posterior[:, 1] = h(Z.dot(w.reshape(-1, 1))).ravel()
#     eps = 1e-15
#     approx_posterior[(approx_posterior[:, 1] == 1), 1] -= eps
#     approx_posterior[(approx_posterior[:, 1] == 0), 1] += eps
    
#     # class-0: P(L=0) = 1 - h(w*x)
#     approx_posterior[:, 0] = 1 - approx_posterior[:, 1]
    
#     discriminant_score = approx_posterior[:, 1] / approx_posterior[:, 0]
    
#     decision = (discriminant_score > 1)
    
#     ind00 = ~decision & (y == 0)
#     ind01 = ~decision & (y == 1)
#     ind10 = decision & (y == 0)
#     ind11 = decision & (y == 1)
    
#     plt.figure(figsize=[4, 4], dpi=150)
#     plt.scatter(X[ind00, 0], X[ind00, 1], marker='o', color='g', alpha=.1, 
#                 label='L0: correct')
#     plt.scatter(X[ind01, 0], X[ind01, 1], marker='s', color='r', alpha=.1,
#                 label='L1: wrong')
#     plt.scatter(X[ind10, 0], X[ind10, 1], marker='o', color='r', alpha=.1,
#                 label='L0: wrong')
#     plt.scatter(X[ind11, 0], X[ind11, 1], marker='s', color='g', alpha=.1,
#                 label='L1: correct')
#     plt.xlabel(r'$x_1$')
#     plt.ylabel(r'$x_2$')
#     plt.title('Decisions by approximate posterior for ' + 
#               r'$D_{{{}}}^{{{}}}$'.format(dataset_name[0], dataset_name[1]), 
#               fontsize=8)
#     plt.legend(fontsize=6)
#     plt.tight_layout()
#     plt.axis('equal')
#     plt.show()
    
#     return discriminant_score


def regression_score(w, X, Type):
    '''get score of regression'''
    
    N, d = X.shape
    
    # ensure that w is a vector
    w = w.reshape(-1, 1)
    
    if (Type == 'linear'):
        Z = np.ones((N, d+1))
        Z[:, 1:] = X
    elif (Type == 'quadratic'):
        Z = np.ones((N, d+4))
        Z[:, 1:d+1] = X
        Z[:, d+1] = X[:, 0]**2
        Z[:, d+2] = X[:, 0] * X[:, 1]
        Z[:, d+3] = X[:, 1]**2
    
    return Z.dot(w)


def costfunction(w, X, y, Type):
    '''formulate the cost value to be minimized'''
    
    N = y.size
    
    # logistic-linear-function
    h = lambda x: 1 / (1 + np.exp(-x))
    
    reg_score = regression_score(w, X, Type)
    
    cost_value = -(1 / N) * (y * np.log(h(reg_score)) + \
                         (1 - y) * np.log(1 - h(reg_score))).sum()
     
    return float(cost_value)


def regression_part3(X_train, y_train, X_test, y_test):
    '''linear and quadratic regression using logistic general function'''
    
    # search for weight vector of linear regression
    res_linear = minimize(fun=costfunction, x0=np.ones(3)*.5, 
                           args=(X_train, y_train, 'linear'), tol=1e-4)
    
    w_linear = res_linear.x

    decision_linear = (regression_score(w_linear, X_test, 'linear') > 0).ravel()
    
    # search for weight vector of quadratic regression
    res_quad = minimize(fun=costfunction, x0=np.ones(6)*.1, 
                           args=(X_train, y_train, 'quadratic'), tol=1e-4)
    
    w_quad = res_quad.x

    decision_quad = (regression_score(w_quad, X_test, 'quadratic') > 0).ravel()
    
    return w_linear, decision_linear, w_quad, decision_quad


def plotregression(X_test, y_test, w_linear, decision_linear, w_quad, 
                   decision_quad, trainset_name, figname):
    
    # linear boundary
    x11 = np.array([min(X_test[:, 0]), max(X_test[:, 0])])
    x22 = -1 / w_linear[2] * (w_linear[0] + w_linear[1] * x11)
    
    ind00 = ~decision_linear & (y_test == 0)
    ind01 = ~decision_linear & (y_test == 1)
    ind10 = decision_linear & (y_test == 0)
    ind11 = decision_linear & (y_test == 1)
    
    plt.figure(figsize=[3, 3], dpi=150)
    plt.scatter(X_test[ind00, 0], X_test[ind00, 1], marker='o', color='g', 
                alpha=.1, facecolors='none', label='L0: correct')
    plt.scatter(X_test[ind01, 0], X_test[ind01, 1], marker='+', color='r', 
                alpha=.1, label='L1: wrong')
    plt.scatter(X_test[ind10, 0], X_test[ind10, 1], marker='o', color='r', 
                alpha=.1, facecolors='none', label='L0: wrong')
    plt.scatter(X_test[ind11, 0], X_test[ind11, 1], marker='+', color='g', 
                alpha=.1, label='L1: correct')
    plt.plot(x11, x22, label='Boundary')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Training based on ' + 
              r'$D_{{{0}}}^{{{1}}}$'.format(trainset_name[0], trainset_name[1]))
    plt.legend(fontsize=5)
    plt.tight_layout()
    plt.savefig('../hw2/reg_linear_' + figname + '.pdf')
    plt.show()

    
    # quadratic boundary
    x1 = np.linspace(min(X3[:, 0]) - 2, max(X3[:, 0]) + 2)
    x2 = np.linspace(min(X3[:, 1]) - 2, max(X3[:, 1]) + 2)
    x1grid, x2grid = np.meshgrid(x1, x2, indexing='ij')
    
    boundary_score = np.zeros((x1.size, x2.size))
    for row, i in enumerate(x1):
        for col, j in enumerate(x2):
            temp = np.array([1, i, j, i**2, i*j, j**2])
            boundary_score[row, col] = (w_quad * temp).sum()
            
    ind00 = ~decision_quad & (y_test == 0)
    ind01 = ~decision_quad & (y_test == 1)
    ind10 = decision_quad & (y_test == 0)
    ind11 = decision_quad & (y_test == 1)
    
    plt.figure(figsize=[3, 3], dpi=150)
    plt.scatter(X_test[ind00, 0], X_test[ind00, 1], marker='o', color='g', 
                alpha=.1, facecolors='none', label='L0: correct')
    plt.scatter(X_test[ind01, 0], X_test[ind01, 1], marker='+', color='r', 
                alpha=.1, label='L1: wrong')
    plt.scatter(X_test[ind10, 0], X_test[ind10, 1], marker='o', color='r', 
                alpha=.1, facecolors='none', label='L0: wrong')
    plt.scatter(X_test[ind11, 0], X_test[ind11, 1], marker='+', color='g', 
                alpha=.1, label='L1: correct')
    plt.contour(x1grid, x2grid, boundary_score, label='Boundary')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Training based on ' + 
              r'$D_{{{0}}}^{{{1}}}$'.format(trainset_name[0], trainset_name[1]))
    plt.legend(fontsize=5)
    plt.savefig('../hw2/reg_quad_' + figname + '.pdf')
    plt.tight_layout()
    plt.show()

    return



# Part3.a: training on 10k samples

# plot decision on training set
w3_L, decision3_L, w3_Q, decision3_Q = regression_part3(X3, y3, X3, y3)
plotregression(X3, y3, w3_L, decision3_L, w3_Q, 
                    decision3_Q, ['train', '10k'], 'train3')

# plot decision on test set
w34_L, decision34_L, w34_Q, decision34_Q = regression_part3(X3, y3, X4, y4)
plotregression(X4, y4, w34_L, decision34_L, w34_Q, 
                    decision34_Q, ['train', '10k'], 'test3')


# Part3.b: training on 1000 samples

# plot decision on training set
w2_L, decision2_L, w2_Q, decision2_Q = regression_part3(X2, y2, X2, y2)
plotregression(X2, y2, w2_L, decision2_L, w2_Q, 
                    decision2_Q, ['train', '1000'], 'train2')

# plot decision on test set
w24_L, decision24_L, w24_Q, decision24_Q = regression_part3(X2, y2, X4, y4)
plotregression(X4, y4, w24_L, decision24_L, w24_Q, 
                    decision24_Q, ['train', '1000'], 'test2')

# Part3.c: training on 100 samples

# plot decision on training set
w1_L, decision1_L, w1_Q, decision1_Q = regression_part3(X1, y1, X1, y1)
plotregression(X1, y1, w1_L, decision1_L, w1_Q, 
                    decision1_Q, ['train', '100'], 'train1')

# plot decision on test set
w14_L, decision14_L, w14_Q, decision14_Q = regression_part3(X1, y1, X4, y4)
plotregression(X4, y4, w14_L, decision14_L, w14_Q, 
                    decision14_Q, ['train', '100'], 'test1')