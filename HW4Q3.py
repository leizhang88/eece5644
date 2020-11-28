import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from PIL import Image

def image_data(filename):
    '''read image and convert to a 5-column array'''

    image = Image.open(filename)
    data_temp = np.asarray(image)
    n_row, n_col = data_temp.shape[0:2]

    data = np.zeros((n_row * n_col, 5))
    for i, slice in enumerate(data_temp):
        temp = np.zeros((n_col, 5))
        temp[:, 0] = i * np.ones(n_col)
        temp[:, 1] = np.arange(n_col)
        temp[:, 2::] = slice
        data[i*n_col:(i+1)*n_col, :] = temp

    # normalize all features to range [0, 1]
    data = normalize(data, norm='max', axis=0)
    return n_row, n_col, data

def gmm_cluster(data, n_class):
    '''training a cluster with Gaussian mixture model'''

    gmm = GaussianMixture(n_components=n_class,
                          covariance_type='full',
                          tol=1e-3,
                          max_iter=500,
                          n_init=5,
                          init_params='kmeans').fit(data)
    return gmm

def draw_clustering(gmm, data, n_row, n_col, figname):
    '''draw the result of clustering, using Gaussian mixture components as class pdfs'''

    alpha = gmm.weights_
    n_components = alpha.size

    class_posterior = np.zeros((data.shape[0], n_components))
    for i in range(n_components):
        class_posterior[:, i] = multivariate_normal.pdf(data, mean=gmm.means_[i, :],
                                                        cov=gmm.covariances_[i, :, :]) * alpha[i]
    labels = np.argmax(class_posterior, axis=1)

    plt.imshow(labels.reshape(n_row, n_col))
    plt.savefig('../hw4/cluster_' + figname + '.pdf')
    plt.close()
    return

def model_selection(data, m_max, K=10):
    '''use K-fold cross-validation to select the optimal number of components for a Gaussian mixture'''

    r, c = data.shape
    scores = np.zeros(m_max)

    for m in tqdm(range(m_max), ascii=True, desc='GMM order selection'):
        test_score = 0
        #for i in range(K):
        #    data_split = np.array_split(data, K)
        #    data_test = data_split[i]
        #    data_train = np.concatenate(data_split.pop(i), axis=None).reshape(-1, c)
        #    # gmm model with (m+1) components
        #    gmm = gmm_cluster(data_train, m+1)
        #    test_score += np.sum(gmm.score_samples(data_test))
        kf = KFold(n_splits=K, shuffle=False).split(data)
        for train_ind, test_ind in kf:
            data_train = data[train_ind, :]
            data_test = data[test_ind, :]
            # gmm model with m+1 components
            gmm = gmm_cluster(data_train, m+1)
            test_score += np.sum(gmm.score_samples(data_test))

        scores[m] = test_score / r

    return scores


files = ['../hw4/3096_color.jpg', '../hw4/42049_color.jpg']
m_max = 10
K = 10

scores = np.zeros((len(files), m_max))
for i, file in enumerate(files):
    n_row, n_col, data = image_data(file)
    gmm_2 = gmm_cluster(data, 2)
    draw_clustering(gmm_2, data, n_row, n_col, '2_img'+str(i))

    scores[i, :] = model_selection(data, m_max, K)
    selected_m = np.argmax(scores[i, :]) + 1

    gmm_selected = gmm_cluster(data, selected_m)
    draw_clustering(gmm_selected, data, n_row, n_col, 'selected_img'+str(i))


plt.rcParams.update({'font.size': 14})
plt.plot(np.arange(m_max)+1, scores[0, :], marker='.', label='plane image')
plt.plot(np.arange(m_max)+1, scores[1, :], marker='.', label='bird image')
plt.scatter(np.argmax(scores[0, :])+1, scores[0, :].max(), s=40, edgecolors='r', facecolors='none')
plt.scatter(np.argmax(scores[1, :])+1, scores[1, :].max(), s=40, edgecolors='r', facecolors='none')
plt.xlabel('n_components')
plt.ylabel('average validation log-likelihood')
plt.title('Model order selection', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('../hw4/order_selection_q3.pdf')
plt.close()

with open('../hw4/res_q3.txt', 'w') as f:
    f.write('# average validation log-likelihood\n')
    np.savetxt(f, scores, fmt='%10.4f')

    f.write('# selected Gaussian components\n')
    np.savetxt(f, np.argmax(scores, axis=1)+1, fmt='%6.0f')

