import scipy.io
data = scipy.io.loadmat('NewsGroup.mat')
TRAIN_LABEL = data['TRAIN_LABEL']
TEST_LABEL = data['TEST_LABEL']
import numpy as np
split_TEST_DATA = np.load("split_TEST_DATA.npy")
split_TRAIN_DATA = np.load("split_TRAIN_DATA.npy")
print np.shape(split_TEST_DATA)
print np.shape(split_TRAIN_DATA)
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB(alpha=1,fit_prior = True)
y_pred = mnb.fit(split_TRAIN_DATA,np.reshape(TRAIN_LABEL,(np.shape(TRAIN_LABEL)[0])))
print(mnb.score(split_TRAIN_DATA,np.reshape(TRAIN_LABEL,(np.shape(TRAIN_LABEL)[0]))))
print(mnb.score(split_TEST_DATA,np.reshape(TEST_LABEL,(np.shape(TEST_LABEL)[0]))))

from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB(alpha=1,fit_prior = True)
y_pred = bnb.fit(split_TRAIN_DATA,np.reshape(TRAIN_LABEL,(np.shape(TRAIN_LABEL)[0])))
print(bnb.score(split_TRAIN_DATA,np.reshape(TRAIN_LABEL,(np.shape(TRAIN_LABEL)[0]))))
print(bnb.score(split_TEST_DATA,np.reshape(TEST_LABEL,(np.shape(TEST_LABEL)[0]))))