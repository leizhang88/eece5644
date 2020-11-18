'''
EECE5644, homework3, question1

Approximate class posteriors with MLP
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# initiate random state
np.random.seed(1234)

d = 3   # data dimension
Ntrain = [100, 200, 500, 1e3, 2e3, 5e3]
Ntest = 1e5
NC = 4  # number of class
K = 10  # K-fold
class_prior = np.ones(NC) / NC
means = np.array([[1, 1, 1],
                  [5, 2, 3],
                  [3, 3, 1],
                  [1, 4, 4]])
covs = np.zeros((NC, d, d))
for i in range(NC):
    temp = np.random.random(d) * 3
    covs[i, :, :] = np.diag(temp)

# print('covs: ', covs)

def generate_samples(N, class_prior, means, covs):
    '''draw samples from given Gaussian pdfs'''

    N = int(N)
    NC = class_prior.size
    cum_class_prior = np.insert(np.cumsum(class_prior), 0, 0)

    temp = np.random.random(N)
    y = np.zeros(N)
    X = np.zeros((N, covs.shape[1]))

    for i in range(NC):
        ind = (temp >= cum_class_prior[i]) & (temp < cum_class_prior[i+1])
        y[ind] = i + 1
        X[ind, :] = np.random.multivariate_normal(mean=means[i, :], cov=covs[i, :, :], size=ind.sum())

    return X, y

# fig = plt.figure()
# # ax = fig.gca()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='Accent')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.show()

def p_error(label_test, decis_test, label_train, NC, true_pdf=False):
    p_error = 0
    for c in range(NC):
        if true_pdf:
            class_prior = 1 / NC
        else:
            class_prior = (label_train == c + 1).sum() / label_train.size

        if (label_test == c+1).sum() == 0:
            continue
        else:
            p_error += ((label_test == c+1) & (decis_test != c+1)).sum() / (label_test == c+1).sum() * class_prior

    return p_error

# ----------------------------------------------
# generate samples
# ----------------------------------------------
# train dataset
traindata_X = dict()
traindata_y = dict()
for i, N in enumerate(Ntrain):
    temp_X, temp_y = generate_samples(N, class_prior, means, covs)
    traindata_X[str(i)] = temp_X
    traindata_y[str(i)] = temp_y

# test dataset
testdata_X, testdata_y = generate_samples(Ntest, class_prior, means, covs)


# MAP classifier with true pdfs for the test dataset
class_posterior = np.zeros((int(Ntest), NC))
for c in range(NC):
    class_posterior[:, c] = multivariate_normal.pdf(testdata_X, mean=means[c, :], cov=covs[c, :, :]) * class_prior[c]
class_posterior /= class_posterior.sum(axis=1).reshape(-1, 1)

expected_risk = class_posterior.dot(np.ones((NC, NC)) - np.eye(NC))
decision = np.argmin(expected_risk, axis=1) + 1

min_perror = p_error(testdata_y, decision, testdata_y, NC, true_pdf=True)
print('Theoretically minimum probability of error: {:>6.3f}'.format(min_perror))
print()


#X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=.1)
#clf = MLPClassifier(hidden_layer_sizes=10, activation='relu', solver='adam', max_iter=1000).fit(X_train, y_train)

# ----------------------------------------------
# Model selection
# ----------------------------------------------
def select_model(X, y, K, NC):
    '''use K-fold cross-validation to select the best hidden layer size'''

    N, d = X.shape
    val_perror = np.zeros(0)
    val_p = np.zeros(0)
    keep_search = True
    count_no_improve = 0
    p = 0

    while keep_search:
        p += 5
        cum_perror = 0
        for k in range(K):
            tempx = np.array_split(X, K)
            tempy = np.array_split(y, K)
            X_test = tempx[k]
            X_train = np.concatenate(tempx.pop(k), axis=None).reshape(-1, d)
            y_test = tempy[k]
            y_train = np.concatenate(tempy.pop(k), axis=None)
            clf = MLPClassifier(hidden_layer_sizes=p,
                                activation='relu',
                                solver='adam',
                                tol=1e-3,
                                max_iter=1000).fit(X_train, y_train)
            decision = clf.predict(X_test)
            # cum_perror += y_test.size * clf.score(X_test, y_test)
            cum_perror += y_test.size * p_error(y_test, decision, y_train, NC)

        new_perror = cum_perror / y.size
        if val_perror.size == 0:
            pass
        else:
            if new_perror - np.min(val_perror) > -1e-3:
                count_no_improve += 1
            else:
                count_no_improve = 0
        # print(new_perror, count_no_improve)

        val_perror = np.append(val_perror, new_perror)
        val_p = np.append(val_p, p)

        if count_no_improve == 5:
            keep_search = False

    best_p = val_p[np.argmin(val_perror)]

    plt.rcParams.update({'font.size': 14})
    plt.scatter(best_p, val_perror.min(), s=40, edgecolors='r', facecolors='none', label='selected model')
    plt.plot(val_p, val_perror, marker='.')
    plt.xlabel('hidden layer size')
    plt.ylabel(r'cumulative $P_{error}$')
    plt.title('N = {}'.format(N), fontsize=16, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../hw3/model_selection_' + str(N) + '.pdf')
    plt.close()

    return best_p

# best_p_test = select_model(testdata_X, testdata_y, K, NC)
# print('best p for test dataset: {:>4.0f}'.format(best_p_test))

best_p = np.zeros_like(Ntrain)
for i in tqdm(range(len(Ntrain)), ascii=True, desc='model selection'):
    best_p[i] = select_model(traindata_X[str(i)], traindata_y[str(i)], K, NC)

# ----------------------------------------------
# Model training and performance assessment
# ----------------------------------------------
test_perror = np.zeros_like(Ntrain)
for i in range(len(Ntrain)):
    clf = MLPClassifier(hidden_layer_sizes=int(best_p[i]),
                        activation='relu',
                        solver='adam',
                        tol=1e-3,
                        max_iter=1000).fit(traindata_X[str(i)], traindata_y[str(i)])
    decision = clf.predict(testdata_X)
    test_perror[i] = p_error(testdata_y, decision, traindata_y[str(i)], NC)

plt.rcParams.update({'font.size': 14})
plt.semilogx(Ntrain, np.ones_like(Ntrain) * min_perror, 'r--', label='theoretically minimum')
plt.semilogx(Ntrain, test_perror, marker='.', label='trained performance')
plt.xlabel('sample size N')
plt.ylabel(r'$P_{error}$ for the test dataset')
plt.title('Performance Assessment', fontsize=16, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('../hw3/test_score.pdf')
plt.close()

# save results in a file
with open('../hw3/result.txt', 'w') as f:
    f.write('# selected hidden layer sizes\n')
    np.savetxt(f, best_p, fmt='%4d')

    f.write('# probability of error for the test dataset\n')
    np.savetxt(f, test_perror, fmt='%10.4f')

    f.write('# theoretically minimum probability of error\n')
    np.savetxt(f, np.array([min_perror]), fmt='%10.4f')


