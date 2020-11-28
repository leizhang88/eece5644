import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn import svm
from tqdm import tqdm
import scipy.io

data = scipy.io.loadmat('../hw4/data_q2.mat')
X_train = data['X_train'].T
y_train = data['y_train'].ravel()
X_test = data['X_test'].T
y_test = data['y_test'].ravel()

plt.rcParams.update({'font.size': 14})
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='.')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Training data', fontsize=16)
plt.tight_layout()
plt.axis('equal')
plt.savefig('../hw4/samples_q2.pdf')
plt.close()

K = 10
C_list = np.logspace(-3, 3, 10)
gamma_list = np.logspace(-2, 4, 12)

# use grid-search for hyper-parameter optimization
grid_search = GridSearchCV(estimator=svm.SVC(),
                           param_grid={'C': C_list,
                                       'gamma': gamma_list},
                           cv=KFold(n_splits=K, shuffle=True, random_state=1234),
                           scoring='accuracy')
grid_search.fit(X_train, y_train)
test_scores = grid_search.cv_results_['mean_test_score']

selected_params = grid_search.best_params_
print('best params:', selected_params)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xx, yy = np.meshgrid(C_list, gamma_list)
ax.scatter(np.log10(xx), np.log10(yy), test_scores.reshape(gamma_list.size, -1), '.')
ax.scatter(np.log10(selected_params['C']), np.log10(selected_params['gamma']), test_scores.min(),
           s=40, marker='o', c='r', label='selected model')
ax.set_xlabel(r'$log_{10}(C)$')
ax.set_ylabel(r'$log_{10}(\gamma)$')
ax.set_zlabel('mean test score')
# ax.set_title('Hyper-parameter optimization', fontsize=16)
plt.tight_layout()
ax.view_init(elev=20, azim=-150)
plt.savefig('../hw4/order_selection_q2.pdf')
plt.close()

# training a SVC model with training data and apply it to the test data
clf = svm.SVC(C=selected_params['C'],
              kernel='rbf',
              gamma=selected_params['gamma']).fit(X_train, y_train)
y_predict = clf.predict(X_test)

ind00 = (y_predict == 1) & (y_test == 1)
ind01 = (y_predict == 1) & (y_test == 2)
ind10 = (y_predict == 2) & (y_test == 1)
ind11 = (y_predict == 2) & (y_test == 2)
p_error = (ind01.sum() + ind10.sum()) / y_test.size
print('test probability of error:', p_error)

plt.scatter(X_test[ind00, 0], X_test[ind00, 1], marker='.', c='g', alpha=.5, label='correct class 1')
plt.scatter(X_test[ind10, 0], X_test[ind10, 1], marker='.', c='r', alpha=.5, label='incorrect class 1')
plt.scatter(X_test[ind11, 0], X_test[ind11, 1], marker='+', c='g', alpha=.5, label='correct class 2')
plt.scatter(X_test[ind01, 0], X_test[ind01, 1], marker='+', c='r', alpha=.5, label='incorrect class 2')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Prediction for the test data')
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.savefig('../hw4/predict_q2.pdf')
plt.close()

with open('../hw4/res_q2.txt', 'w') as f:
    f.write('# C list\n')
    np.savetxt(f, C_list, fmt='%10.4e')

    f.write('# gamma list\n')
    np.savetxt(f, gamma_list, fmt='%10.4e')

    f.write('# best params: C, gamma\n')
    np.savetxt(f, np.array([selected_params['C'], selected_params['gamma']]), fmt='%10.4e')

    f.write('# error probability for test data\n')
    np.savetxt(f, np.array([p_error]), fmt='%10.4f')

