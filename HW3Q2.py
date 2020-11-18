'''
EECE5644, homework3, question2

Use k-fold cross-validation to select hyper-parameter
'''

import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm

# initiate random state
np.random.seed(2345)

d = 7
Ntrain = 100
Ntest = 10000
K = 10
true_mu = np.random.random(d)
temp = np.random.random((d, 1))
true_cov = temp.dot(temp.T) + np.eye(d) * 0.1
true_w = np.random.random(d)
true_alpha = np.diag(true_cov).sum() / d * np.array([1e-3, 1e-2, 1e-1, 1, 1e1, 2e1])


def generate_samples(N, d, mu, cov, weight, x_noise_sigma):
    '''draw samples from a Gaussian model with additive white noise in both input and output'''

    # input noise
    x_noise = np.random.multivariate_normal(mean=np.zeros(d), cov=x_noise_sigma*np.eye(d), size=N)

    # output noise
    y_noise = np.random.random(N)

    # input
    X = np.random.multivariate_normal(mean=mu, cov=cov, size=N)

    # output
    y = (np.dot(X + x_noise, weight.reshape(-1, 1))).ravel() + y_noise

    return X, y


def linear_model_training(X, y, w0, beta):
    '''
    Estimate the linear weight vector for data with
    additive white noise to the output using MAP;
    The weight vector prior is a Gaussian with mean=0, cov=beta*I
    '''

    def loss_fun(w, X, y, beta):
        '''compute loss'''

        N, d = X.shape
        assert len(w) == d + 1, "shape of weight vector is incompatible with X"

        Z = np.ones((N, d + 1))
        Z[:, 1::] = X

        loss = ((y - Z.dot(w.reshape(-1, 1)).ravel())**2).sum() / (2 * N) - \
            np.log(multivariate_normal.pdf(w, mean=np.zeros(d+1), cov=beta*np.eye(d+1)))

        return loss

    res = minimize(loss_fun, w0, args=(X, y, beta), tol=1e-6)

    return res.x


def linear_model_analytical(X, y, beta):
    '''return analytical solution of the linear weight vector'''

    N, d = X.shape
    Z = np.ones((N, d + 1))
    Z[:, 1::] = X

    w = np.linalg.inv(np.dot(Z.T, Z) + 1/beta * np.eye(d + 1)).dot(np.dot(Z.T, y.reshape(-1, 1)))
    return w


def select_hyper_param(X, y, K, alpha, figname):
    '''Select hyper parameter 'beta' for the weight vector prior, using K-fold cross-validation'''

    N, d = X.shape
    scores = np.zeros(0)
    betas = np.zeros(0)
    beta = 0.1
    continue_search = True
    count_no_improve = 0

    while continue_search:
        score = 0
        for j in range(K):
            temp_X = np.array_split(X, K)
            temp_y = np.array_split(y, K)
            X_cvtest = temp_X[j]
            X_cvtrain = np.concatenate(temp_X.pop(j), axis=None).reshape(-1, d)
            y_cvtest = temp_y[j]
            y_cvtrain = np.concatenate(temp_y.pop(j), axis=None)

            w = linear_model_training(X_cvtrain, y_cvtrain, np.ones(d+1), beta)

            Z_cvtest = np.ones((X_cvtest.shape[0], X_cvtest.shape[1] + 1))
            Z_cvtest[:, 1::] = X_cvtest
            score += y_cvtest.size * 0.5 * ((y_cvtest - Z_cvtest.dot(w.reshape(-1, 1)).ravel())**2).sum()

        new_score = score / N
        if scores.size == 0:
            pass
        else:
            if new_score - scores.min() > -.5:
                count_no_improve += 1
            else:
                count_no_improve = 0

        scores = np.append(scores, new_score)
        betas = np.append(betas, beta)

        beta *= 2

        if count_no_improve == 5 or betas.size > 100:
            continue_search = False

    best_beta = betas[np.argmin(scores)]
    plt.rcParams.update({'font.size': 14})
    plt.semilogx(betas, scores, marker='.')
    plt.scatter(best_beta, scores.min(), s=40, marker='o', edgecolors='r', facecolors='none',
                label='selected beta')
    plt.title(r'$\alpha =$ {:.3e}'.format(alpha))
    plt.xlabel('beta')
    plt.ylabel('minus log-likelihood')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../hw3/beta_search_' + figname + '.pdf')
    plt.close()

    return best_beta, betas.size

# tempalpha = np.diag(true_cov).sum() / d * 100
# X_train, y_train = generate_samples(Ntrain, d, true_mu, true_cov, true_w, tempalpha)
# X_test, y_test = generate_samples(Ntest, d, true_mu, true_cov, true_w, tempalpha)
# w1 = linear_model_training(X_train, y_train, np.ones(d+1), .1)
# w2 = linear_model_analytical(X_train, y_train, .1)
# print('w1:', w1, 'w2:', w2)
#
# beta1, temp = select_hyper_param(X_train, y_train, K, tempalpha, '_')
# print(beta1, temp)

n_alpha = true_alpha.size
test_score = np.zeros(n_alpha)
best_beta = np.zeros(n_alpha)
betas_size = np.zeros(n_alpha)
for i in tqdm(range(n_alpha), ascii=True, desc='running'):
    # generate data
    X_train, y_train = generate_samples(Ntrain, d, true_mu, true_cov, true_w, true_alpha[i])
    X_test, y_test = generate_samples(Ntest, d, true_mu, true_cov, true_w, true_alpha[i])

    # hyper-parameter optimization
    best_beta[i], betas_size[i] = select_hyper_param(X_train, y_train, K, true_alpha[i], str(i))

    # model optimization
    # w = linear_model_training(X_train, y_train, np.ones(d+1), best_beta[i])
    w = linear_model_analytical(X_train, y_train, best_beta[i])

    # validation for the test dataset
    Z = np.ones((Ntest, d+1))
    Z[:, 1::] = X_test
    test_score[i] = ((y_test - Z.dot(w.reshape(-1, 1)).ravel())**2).sum()

plt.rcParams.update({'font.size': 14})
plt.loglog(true_alpha, test_score, marker='.')
plt.title('Impact of Input Noise')
plt.xlabel(r'input noise parameter $\alpha$')
plt.ylabel(r'$-2 \times$ log-likelihood')
plt.tight_layout()
plt.savefig('../hw3/test_score_q2.pdf')
plt.close()

plt.loglog(true_alpha, best_beta, marker='.')
plt.title('Hyper-parameter v.s. Input Noise')
plt.xlabel(r'input noise parameter $\alpha$')
plt.ylabel(r'hyper-parameter $\beta$')
plt.tight_layout()
plt.savefig('../hw3/betas_q2.pdf')
plt.close()

with open('../hw3/result_q2.txt', 'w') as f:
    f.write('# true mu\n')
    np.savetxt(f, true_mu, fmt='%10.4f')

    f.write('# true covariance matrix\n')
    np.savetxt(f, true_cov, fmt='%10.4f')

    f.write('# true weight vector\n')
    np.savetxt(f, true_w, fmt='%10.4f')

    f.write('# true input noise parameter, alpha\n')
    np.savetxt(f, true_alpha, fmt='%12.6e')

    f.write('# selected hyper-parameter, beta\n')
    np.savetxt(f, best_beta, fmt='%10.4e')

    f.write('# test score\n')
    np.savetxt(f, test_score, fmt='%12.6e')

