'''
EECE5644 Intro2ML - homework2, question2

Application of BIC and k-fold cross-validation
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import random
from sklearn import mixture
from scipy.stats import multivariate_normal
import progressbar
import time
import seaborn as sns

#-----------------
# define functions
#-----------------
# assume covariance matrices spherically symmetric
# cov_value is in shape (M,)
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
        X[ind, :] = random.multivariate_normal(means[:, i].flatten(), 
         Covs[i, :, :], ind.sum())
    
    return X, gmm_label

# evaluate GMM pdf at each row of X
def evaluateGMM(X, alpha, means, Covs):
    N = X.shape[0]
    gmm_pdf = np.zeros(N)
    
    for i in range(alpha.size):
        rv = multivariate_normal(mean=means[:, i].ravel(), 
                                 cov=Covs[i, :, :])
        gmm_pdf = gmm_pdf + alpha[i] * rv.pdf(X)
    
    return gmm_pdf

# plot contour for GMM only when n_feature=2
def contourGMM(X, alpha, means, Covs):
    if (X.shape[1] == 2):
        x1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]))
        x2 = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]))
        x1grid, x2grid = np.meshgrid(x1, x2)
        temp = np.array([x1grid.ravel(), x2grid.ravel()]).T
        Z = evaluateGMM(temp, alpha, means, Covs).reshape(x1grid.shape)
        
    return x1grid, x2grid, Z

#--------------------------------------
# assign values for essential variables
#--------------------------------------
# sample dimension
d = 2

# sample number
Nset = np.array([1e2, 1e3, 1e4, 1e5, 1e6]).astype('int')
#, 1e4, 1e5, 1e6
# number of components for true GMM
trueMset = np.array([15]) 

# max guess of M
maxM = 40 

assert maxM > max(trueMset), 'Try larger \'maxM\''

# number of trials
n_experiment = 50


# specify a set of std, upon which the covariances are valued
# a factor of sqrt(2) is becaus the distance between means is 2*sqrt(2)
rand_std = np.array(
    [0.58, 0.56, 0.09, 0.45, 0.55, 0.5 , 0.35, 0.6 , 0.57, 0.44, 0.27, 
     0.51, 0.46, 0.71, 0.67, 0.26, 0.5 , 0.58, 0.02, 0.14, 0.6 , 0.61,
     0.06, 0.35, 0.46, 0.36, 0.53, 0.43, 0.18, 0.37, 0.55, 0.53, 0.57,
     0.42, 0.47, 0.44, 0.22, 0.7 , 0.15, 0.2 , 0.08, 0.32, 0.26, 0.46,
     0.37, 0.23, 0.26, 0.61, 0.7 , 0.61]) * np.sqrt(2)


#---------------
# start training
#---------------

# set up progressbar
bar = progressbar.ProgressBar(maxval=trueMset.size * n_experiment * \
                              Nset.size, \
                              widgets=[progressbar.Bar('=', '[', ']'), ' ', \
                                       progressbar.Percentage(),'\n'])
# bar.start()

# t_begin = time.time()

# # preallocate space for order-selection results
# MOS_bic = np.zeros((trueMset.size, n_experiment, Nset.size))
# MOS_kcv = np.zeros((trueMset.size, n_experiment, Nset.size))

# for i_M in range(trueMset.size):
#     # current true GMM components
#     trueM = trueMset[i_M]
    
#     # true GMM pdf
    
#     # GMM weights
#     alpha = np.ones(trueM) / trueM
                
#     # assume means evenly spaced on line y = x
#     means = np.ones((d, 1)).dot(np.arange(trueM).reshape(1, -1) * 2)
    
#     # spherically symmetric covariances
#     cov_value = (rand_std[:trueM])**2
    
#     Covs = generateCov(d, cov_value)
    
    
#     for i_E in range(n_experiment):
#         # run the i_Eth training
    
    
#         for i_N in range(Nset.size):
            
#             # current sample number
#             N = Nset[i_N]
            
#             # generate samples from true GMM, shape(N, d)
#             X, dummy = generateGMM(d, N, alpha, means, Covs)
            
            
#             #------------------------------------
#             # Part1: model order selection by BIC
#             #------------------------------------
            
#             # BIC objective value
#             bic = np.zeros(maxM)
            
#             count_no_change_bic = 0
            
#             early_stopping_bic = False
            
#             for m in range(maxM):
#                 # current GMM component number
#                 M = m + 1
                
#                 # train GMM using scikit-learn library
#                 gmm_bic = mixture.GaussianMixture(n_components=M, 
#                                               covariance_type='spherical', 
#                                               tol=1e-3, 
#                                               reg_covar=1e-6, 
#                                               max_iter=500,
#                                               n_init=1)
#                 gmm_bic.fit(X)
    
#                 # number of parameters for GMM
#                 # n_parm for weights: M
#                 # n_parm for means: d * M
#                 # n_parm for spherically symmetric covariances: M
#                 n_parm = M + (d * M) + M
    
#                 # BIC objective value
#                 bic[m] = -2 * gmm_bic.score_samples(X).sum() + \
#                     n_parm * np.log(N)
                
#                 # terminate training early when BIC is not improving
#                 if (m > 0):
#                     if (bic[m] - min(bic[0:m])) > -1e-3:
#                         count_no_change_bic += 1
#                     else:
#                         count_no_change_bic = 0
                
                
#                 if (count_no_change_bic == 5):
                    
#                     # order selected by BIC
#                     MOS_bic[i_M, i_E, i_N] = np.argmin(bic[0:m]) + 1
                    
#                     # early stopping
#                     early_stopping_bic = True
                    
#                     break
            
#             if not early_stopping_bic:
#                 MOS_bic[i_M, i_E, i_N] = np.argmin(bic) + 1
#             # exit BIC training
            
            
#             #--------------------------------------------------------
#             # Part2: model order selection by K-fold cross-validation
#             #--------------------------------------------------------
            
#             # value of k
#             # k = np.int(np.sqrt(N))
#             k = 5
            
#             # average log-likelihood for each order
#             kcv_score = np.zeros(maxM)
            
#             count_no_change_kcv = 0
            
#             early_stopping_kcv = False
            
#             for m2 in range(maxM):
#                 # current order
#                 M2 = m2 + 1
                
#                 # weighted log-likelihood for each validation set
#                 score_validatset = np.zeros(k) 
                
#                 for i in range(k):
#                     sample_kfold = np.array_split(X, k, axis=0)
        
#                     sample_validate = sample_kfold[i]
        
#                     sample_train = sample_kfold
#                     sample_train.pop(i)
#                     sample_train = np.concatenate(sample_train, 
#                                                   axis=None).reshape(-1, d)
                    
#                     gmm_kcv = mixture.GaussianMixture(n_components=M2, 
#                                                       covariance_type=
#                                                       'spherical', 
#                                                       tol=1e-3, 
#                                                       reg_covar=1e-6, 
#                                                       max_iter=500,
#                                                       n_init=1)
#                     gmm_kcv.fit(sample_train)
        
#                     # weighted by entries in validate set
#                     score_validatset[i] = -1 / N * sample_validate.shape[0] *\
#                         gmm_kcv.score_samples(sample_validate).sum()
                
                
#                 # average log-likelihood for curremt M
#                 kcv_score[m2] = score_validatset.sum() / k
            
#                 if (m2 > 0):
#                     if (kcv_score[m2] - min(kcv_score[0:m2])) > -1e-3:
#                         count_no_change_kcv += 1
#                     else:
#                         count_no_change_kcv = 0
                
#                 if (count_no_change_kcv == 5):
                    
#                     # order selected by K-fold cross-validation
#                     MOS_kcv[i_M, i_E, i_N] = np.argmin(kcv_score[0:m2]) + 1
                    
#                     # early stopping
#                     early_stopping_kcv = True
                    
#                     break
                
            
#             if not early_stopping_kcv:
#                 MOS_kcv[i_M, i_E, i_N] = np.argmin(kcv_score) + 1
            
#             # exit k-fold cross-validation training
            
#             # update progressbar
#             bar.update((i_M + 1) * (i_E + 1) * (i_N + 1))
#             time.sleep(.1)
            
#         # end i_N-loop
        
#     # end i_E-loop
    
# # end i_M-loop

# bar.finish()
    
# t_end = time.time()

# print('[Running time] {:.1} sec'.format(t_end - t_begin)) 

# # x1grid, x2grid, Z = contourGMM(X, gmm.weights_, gmm.means_.T, 
# #                                gmm_covs)

# # plt.figure(figsize=[4, 4], dpi=150)
# # plt.contour(x1grid, x2grid, Z)
# # plt.scatter(X[:, 0], X[:, 1], s=5, c=gmm.predict(X), cmap='viridis')
# # plt.axis('equal')
# # plt.show()

# # write order-selection results in files
# with open('../hw2/q2_bic.txt', 'w') as f:
#     f.write('# Model order selection results using BIC\n' + \
#             '# Results ordered in shape(n_trueM, n_experiment, n_N)\n')
    
#     for data_slice, m in zip(MOS_bic, trueMset):
#         # write out true n_components of GMM for current slice
#         f.write('# True component number is: {:4d}\n'.format(m))
        
#         np.savetxt(f, data_slice, fmt='%-4d')


# with open('../hw2/q2_kcv.txt', 'w') as f:
#     f.write('# Model order selection results using K-fold CV\n' + \
#             '# Results ordered in shape(n_trueM, n_experiment, n_N)\n')
    
#     for data_slice, m in zip(MOS_kcv, trueMset):
#         # write out true n_components of GMM for current slice
#         f.write('# True component number is: {:4d}\n'.format(m))
        
#         np.savetxt(f, data_slice, fmt='%-4d')


#-------------------------------------------
# plot result
#-------------------------------------------

# read results from files
res_bic = np.loadtxt('../hw2/q2_bic.txt')
res_kcv = np.loadtxt('../hw2/q2_kcv.txt')


plt.figure(figsize=[6, 3], dpi=150)
ax = sns.boxplot(data=res_bic, palette='pastel', linewidth=.5, 
                 flierprops=dict(markerfacecolor='.6', markersize=5, alpha=.6))
ax = sns.swarmplot(data=res_bic, color='.2', size=2.5)
ax.set_xticklabels(['1e2', '1e3', '1e4', '1e5', '1e6'])
ax.set_xlabel('Sample number')
ax.set_ylabel('Occurances')
ax.set_title('Results for BIC ($M_{true}=15$, $E=50$)')
plt.tight_layout()
plt.savefig('../hw2/res_bic.pdf')
plt.show()


plt.figure(figsize=[6, 3], dpi=150)
ax = sns.boxplot(data=res_kcv, palette='pastel', linewidth=.5, 
                 flierprops=dict(markerfacecolor='.6', markersize=5, alpha=.6))
ax = sns.swarmplot(data=res_bic, color='.2', size=2.5)
ax.set_xticklabels(['1e2', '1e3', '1e4', '1e5', '1e6'])
ax.set_xlabel('Sample number')
ax.set_ylabel('Occurances')
ax.set_title('Results for K-fold CV ($M_{true}=15$, $E=50$)')
plt.tight_layout()
plt.savefig('../hw2/res_kcv.pdf')
plt.show()







