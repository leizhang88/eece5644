import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

K = 10
N_train = 1000
N_test = 10000

plt.rcParams.update({'font.size': 14})

def generate_sample(N):
    x = np.random.gamma(3, 2, N)
    z = np.exp(x**2 * np.exp(-x/2))
    v = np.random.lognormal(0, .1, N)
    y = v * z
    return x, y

x_train, y_train = generate_sample(N_train)
x_test, y_test = generate_sample(N_test)

plt.scatter(x_train, y_train, marker='.')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Training data', fontsize=16)
plt.tight_layout()
plt.savefig('../hw4/samples_q1.pdf')
plt.close()

continue_search = True
count_no_improve = 0
score_val = np.zeros(0)
p_val = np.zeros(0)
p = 0

# select optimal hidden layer size using K-fold cross-validation
while continue_search:
    p += 5
    validation_score = 0
    for i in range(K):
        tempx = np.array_split(x_train, K)
        tempy = np.array_split(y_train, K)
        x_cv_test = tempx[i].reshape(-1, 1)
        x_cv_train = np.concatenate(tempx.pop(i), axis=None).reshape(-1, 1)
        y_cv_test = tempy[i]
        y_cv_train = np.concatenate(tempy.pop(i), axis=None)

        regr = MLPRegressor(hidden_layer_sizes=p,
                            activation='relu',
                            solver='adam',
                            tol=1e-3,
                            max_iter=2000).fit(x_cv_train, y_cv_train)

        # weighted validation MSE
        validation_score += ((regr.predict(x_cv_test) - y_cv_test)**2).mean() * (y_cv_test.size / N_train)

    if score_val.size == 0:
        pass
    else:
        if validation_score - score_val.min() > -1e-2:
            count_no_improve += 1
        else:
            count_no_improve = 0

    score_val = np.append(score_val, validation_score)
    p_val = np.append(p_val, p)

    if count_no_improve == 5 or p_val.size > 100:
        continue_search = False

best_p = p_val[np.argmin(score_val)]
print('select p:', best_p)
print('select score:', score_val.min())

plt.plot(p_val, score_val, marker='.')
plt.scatter(best_p, score_val.min(), s=40, marker='o', edgecolors='r', facecolors='none',
            label='selected model')
plt.xlabel('hidden layer size')
plt.ylabel('average validation MSE')
plt.title('Model order selection', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('../hw4/order_selection_q1.pdf')
plt.close()

# training a model of selected order with training data
regr = MLPRegressor(hidden_layer_sizes=int(best_p),
                    activation='relu',
                    solver='adam',
                    tol=1e-3,
                    max_iter=2000).fit(x_train.reshape(-1, 1), y_train)
y_test_predict = regr.predict(x_test.reshape(-1, 1))
test_score = ((y_test_predict - y_test)**2).sum() / N_test

plt.scatter(x_test, y_test, marker='.', label='test data')
plt.scatter(x_test, y_test_predict, marker='.', label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predict for the test data', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('../hw4/predict_q1.pdf')
plt.close()

with open('../hw4/res_q1.txt', 'w') as f:
    f.write('# value of p\n')
    np.savetxt(f, p_val, fmt='%6.0f')

    f.write('# value average validation MSE\n')
    np.savetxt(f, score_val, fmt='%10.4f')

    f.write('# selected model: p, MSE\n')
    np.savetxt(f, np.array([best_p, score_val.min()]), fmt='%10.4f')

    f.write('# test data MSE\n')
    np.savetxt(f, np.array([test_score]), fmt='%10.4f')

