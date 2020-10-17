"Take Home Exam 1: Question 1"
import numpy as np
from scipy import linalg, random
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# set random number generator
random.seed(1234)

eps = 1e-16

class_prior = np.array([.7, .3])
NC = class_prior.size

# class parameter, mean
m = np.array([-1 * np.ones((4, 1)), np.ones((4, 1))])

# class parameter, covariance
C = np.array([[[2, -.5, .3, 0],
               [-.5, 1, -.5, 0],
               [.3, -.5, 1, 0],
               [0, 0, 0, 2]],

               [[1, .3, -.2, 0],
               [.3, 2, .3, 0],
               [-.2, .3, 1, 0],
               [0, 0, 0, 3]]])

# sample size
N = 10000
rnd = random.rand(N)
# assign class
y = rnd >= class_prior[0]

# sampling
X = np.zeros((m.shape[1], N))

class_sample_num = np.zeros(NC).astype(int)

temp = np.insert(np.cumsum(class_prior), 0, 0)

for i in range(NC):
    ind = (rnd >= temp[i]) & (rnd < temp[i + 1])
    class_sample_num[i] = np.sum(ind)
    X[:, ind] = random.multivariate_normal(m[i, :, :].flatten(), 
     C[i, :, :], class_sample_num[i]).T

# ----------------------------------------------
# Part A: ERM classification with true class pdf
# ----------------------------------------------
discriminant_score = np.zeros(N)

# likelihood-ratio
for i in range(N):
    discriminant_score[i] = np.log(multivariate_normal.pdf(X[:, i], 
                      mean=m[1, :, :].flatten(), cov=C[1, :, :]) / \
                      multivariate_normal.pdf(X[:, i], 
                                              mean=m[0, :, :].flatten(), 
                                              cov=C[0, :, :]))

# choose threshold as the midpoint of every two sorted discriminant score
score_sort = np.sort(discriminant_score)
threshold = (score_sort[0:-1] + score_sort[1:]) / 2
threshold = np.insert(threshold, 0, score_sort[0] - eps)
threshold = np.append(threshold, score_sort[-1] + eps)

P01 = np.zeros(threshold.size)
P10 = np.zeros(threshold.size)
P11 = np.zeros(threshold.size)
Perr_a = np.zeros(threshold.size)
for i in range(threshold.size):
    ERM_decision = discriminant_score > threshold[i]
    P01[i] = np.sum(~ERM_decision & y) / np.sum(y)
    P10[i] = np.sum(ERM_decision & ~y) / np.sum(~y)
    P11[i] = np.sum(ERM_decision & y) / np.sum(y)
    Perr_a[i] = P01[i] * class_prior[1] + P10[i] * class_prior[0]

# best value for threshold
threshold_best = threshold[np.argmin(Perr_a)]

# plot ROC curve
fig1 = plt.figure(figsize=[4, 3], dpi=150)
ax1 = fig1.add_subplot(111)
ax1.plot(P10, P11, linewidth=1)
ax1.scatter(P10[np.argmin(Perr_a)], P11[np.argmin(Perr_a)], c='r', 
                marker='x', label=r'minimum $P_{error}$')
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_xlabel(r'$P_{FP}$')
ax1.set_ylabel(r'$P_{TP}$')
ax1.set_title('ROC curve with ture class pdf')
ax1.legend()
plt.tight_layout()
plt.savefig('Q1A_roc.pdf')
plt.show()

# plot P_error
fig2 = plt.figure(figsize=[4, 3], dpi=150)
ax2 = fig2.add_subplot(111)
ax2.plot(threshold, Perr_a, linewidth=1)
plt.annotate(f'{min(Perr_a):5.2%}', fontsize =6, 
                xy=(threshold_best - .5, min(Perr_a)),
                xytext=(threshold_best - 15, min(Perr_a)), 
                arrowprops=dict(arrowstyle='->'))
ax2.set_xlabel(r'threshold')
ax2.set_ylabel(r'$P_{error}$')
ax2.set_title(r'$P_{error}$ with true class pdf')
plt.tight_layout()
plt.savefig('Q1A_perror.pdf')
plt.show()


# --------------------------------------------------
# Part B: ERM classification with mismatch class pdf
# --------------------------------------------------
discriminant_score = np.zeros(N)

# likelihood-ratio, change true covariance with identity matrix
for i in range(N):
    discriminant_score[i] = np.log(multivariate_normal.pdf(X[:, i], 
                      mean=m[1, :, :].flatten(), 
                      cov=np.diag(np.diag(C[1, :, :]))) / \
                      multivariate_normal.pdf(X[:, i], 
                      mean=m[0, :, :].flatten(), 
                      cov=np.diag(np.diag(C[0, :, :]))))

# choose threshold as the midpoint of every two sorted discriminant score
score_sort = np.sort(discriminant_score)
threshold_b = (score_sort[0:-1] + score_sort[1:]) / 2
threshold_b = np.insert(threshold_b, 0, score_sort[0] - eps)
threshold_b = np.append(threshold_b, score_sort[-1] + eps)

P01_b = np.zeros(threshold_b.size)
P10_b = np.zeros(threshold_b.size)
P11_b = np.zeros(threshold_b.size)
Perr_b = np.zeros(threshold_b.size)
for i in range(threshold_b.size):
    ERM_decision = discriminant_score > threshold_b[i]
    P01_b[i] = np.sum(~ERM_decision & y) / np.sum(y)
    P10_b[i] = np.sum(ERM_decision & ~y) / np.sum(~y)
    P11_b[i] = np.sum(ERM_decision & y) / np.sum(y)
    Perr_b[i] = P01_b[i] * class_prior[1] + P10_b[i] * class_prior[0]

# best value for threshold
threshold_best = threshold_b[np.argmin(Perr_b)]

# plot ROC curve
fig1 = plt.figure(figsize=[4, 3], dpi=150)
ax1 = fig1.add_subplot(111)
ax1.plot(P10_b, P11_b, linewidth=1)
ax1.scatter(P10_b[np.argmin(Perr_b)], P11_b[np.argmin(Perr_b)], c='r', 
                marker='x', label=r'minimum $P_{error}$')
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_xlabel(r'$P_{FP}$')
ax1.set_ylabel(r'$P_{TP}$')
ax1.set_title('ROC curve with mismatch class pdf')
ax1.legend()
plt.tight_layout()
plt.savefig('Q1B_roc.pdf')
plt.show()

# plot P_error
fig2 = plt.figure(figsize=[4, 3], dpi=150)
ax2 = fig2.add_subplot(111)
ax2.plot(threshold_b, Perr_b, linewidth=1)
plt.annotate(f'{min(Perr_b):5.2%}', fontsize =6, xy=(threshold_best-.5, 
                min(Perr_b)),
                xytext=(threshold_best-10, min(Perr_b)), 
                arrowprops=dict(arrowstyle='->'))
ax2.set_xlabel(r'threshold')
ax2.set_ylabel(r'$P_{error}$')
ax2.set_title(r'$P_{error}$ with mismatch class pdf')
plt.tight_layout()
plt.savefig('Q1B_perror.pdf')
plt.show()


#----------------------------------
# Part C: Fisher LDA classification
#----------------------------------
# class conditional pdf mean and covariance
muhat0 = np.mean(X[:, ~y], axis=1)
sighat0 = np.cov(X[:, ~y])
muhat1 = np.mean(X[:, y], axis=1)
sighat1 = np.cov(X[:, y])

# between class sctter matrix
Sb = np.dot((muhat0 - muhat1).reshape(-1, 1), (muhat0 - muhat1).reshape(1, -1))

# within class scatter matrix
Sw = sighat0 + sighat1

lam, V = linalg.eig(Sb, Sw)

# projection vector is the eigenvector related to the maximum eigenvalue
wLDA = V[:, np.argmax(lam)]

discriminant_score_LDA = np.dot(wLDA.reshape(1, -1), X)

# choose threshold as the midpoint of every two sorted discriminant score
score_sort = np.sort(discriminant_score_LDA).ravel()
threshold_LDA = (score_sort[0:-1] + score_sort[1:]) / 2
threshold_LDA = np.insert(threshold_LDA, 0, score_sort[0] - eps)
threshold_LDA = np.append(threshold_LDA, score_sort[-1] + eps)
    
P01_c = np.zeros(threshold_LDA.size)
P10_c = np.zeros(threshold_LDA.size)
P11_c = np.zeros(threshold_LDA.size)
Perr_c = np.zeros(threshold_LDA.size)
for i in range(threshold_LDA.size):
    LDA_decision = discriminant_score_LDA > threshold_LDA[i]
    P01_c[i] = np.sum(~LDA_decision & y) / np.sum(y)
    P10_c[i] = np.sum(LDA_decision & ~y) / np.sum(~y)
    P11_c[i] = np.sum(LDA_decision & y) / np.sum(y)
    Perr_c[i] = P01_c[i] * class_prior[1] + P10_c[i] * class_prior[0]

# plot ROC curve
fig1 = plt.figure(figsize=[4, 3], dpi=150)
ax1 = fig1.add_subplot(111)
ax1.plot(P10_c, P11_c, linewidth=1)
ax1.scatter(P10_c[np.argmin(Perr_c)], P11_c[np.argmin(Perr_c)], c='r', 
                marker='x', label=r'minimum $P_{error}$')
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_xlabel(r'$P_{FP}$')
ax1.set_ylabel(r'$P_{TP}$')
ax1.set_title('ROC curve by Fisher LDA classifer')
ax1.legend()
plt.tight_layout()
plt.savefig('Q1C_roc.pdf')
plt.show()

# plot P_error
fig2 = plt.figure(figsize=[4, 3], dpi=150)
ax2 = fig2.add_subplot(111)
ax2.plot(threshold_LDA, Perr_c, linewidth=1)
plt.annotate(f'{min(Perr_c):4.2%}', fontsize =6, 
                xy=(threshold_LDA[np.argmin(Perr_c)] - .1, 
                min(Perr_c)),
                xytext=(threshold_LDA[np.argmin(Perr_c)] - 2, min(Perr_c)), 
                arrowprops=dict(arrowstyle='->'))
ax2.set_xlabel(r'threshold')
ax2.set_ylabel(r'$P_{error}$')
ax2.set_title(r'$P_{error}$ by Fisher LDA classifer')
plt.tight_layout()
plt.savefig('Q1C_perror.pdf')
plt.show()

#---------------------------------------------------------------
# Comparisons of ROC curves and P_error between these approaches
#---------------------------------------------------------------
fig3 = plt.figure(figsize=[4, 3], dpi=150)
ax3 = fig3.gca()
ax3.plot(P10, P11, linewidth=1, linestyle='-', c='m', label='ERM true pdf')
ax3.plot(P10_b, P11_b, linewidth=1, linestyle='--', c='g', 
         label='ERM mismatch pdf')
ax3.plot(P10_c, P11_c, linewidth=1, linestyle=':', c='b', label='LDA')
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])
ax3.set_xlabel(r'$P_{FP}$')
ax3.set_ylabel(r'$P_{TP}$')
ax3.set_title('ROC curve comparison')
ax3.legend(fontsize=8)
plt.tight_layout()
plt.savefig('Q1_roccmp1.pdf')
plt.show()

fig4 = plt.figure(figsize=[4, 3], dpi=150)
ax4 = fig4.gca()
ax4.plot(threshold, Perr_a, linewidth=1, linestyle='-', c='m', 
         label='ERM true pdf')
ax4.plot(threshold_b, Perr_b, linewidth=1, linestyle='--', c='g', 
         label='ERM mismatch pdf')
ax4.plot(threshold_LDA, Perr_c, linewidth=1, linestyle=':', c='b', 
         label='LDA')
#ax4.set_ylim([0, .1])
#ax4.set_xlim([-5, 5])
ax4.set_xlabel(r'threshold')
ax4.set_ylabel(r'$P_{error}$')
ax4.set_title(r'$P_{error}$ comparison')
ax4.legend(fontsize=8)
plt.tight_layout()
plt.savefig('Q1_perrcmp1.pdf')
plt.show()
