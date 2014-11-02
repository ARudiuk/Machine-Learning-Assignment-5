from sklearn import svm
import numpy as np
class SVM:
    def __init__(self,kernel='linear',C=1,sigma=1.,degree=1.,threshold=1e-5):
        self.kernel = kernel
        self.C = C
        self.sigma = sigma
        self.degree = degree
        self.threshold = threshold

    def train_svm(self,inputs,targets):
        self.clf = svm.SVC(C=self.C,kernel=self.kernel,degree=self.degree)
#        print self.clf
        self.clf.fit(inputs,targets)
    def test_svm(self,inputs,targets):
#        print "targets",targets,"predicted",self.clf.predict(inputs)
        results =  self.clf.predict(inputs)
        correct = 0
        sum = 0
        for i in range(np.shape(results)[0]):
            if results[i] == targets[i]:
                correct += 1
            sum += 1
        accuracy = (float(correct)/float(sum))*100
        return accuracy

import scipy.io
data = scipy.io.loadmat('NewsGroup.mat')
TRAIN_LABEL = data['TRAIN_LABEL']
TEST_LABEL = data['TEST_LABEL']
split_TEST_DATA = np.load("split_TEST_DATA.npy")
split_TRAIN_DATA = np.load("split_TRAIN_DATA.npy")

errors = np.zeros((21,6))
for i in range(-10,11):    
    print i
    learner = SVM(kernel='linear',C=2**(i))
    learner.train_svm(split_TRAIN_DATA,np.reshape(TRAIN_LABEL,(np.shape(TRAIN_LABEL)[0])))
    errors[i+10,0] = learner.test_svm(split_TEST_DATA,np.reshape(TEST_LABEL,(np.shape(TEST_LABEL)[0])))
    learner = SVM(kernel='rbf',C=2**(i))
    learner.train_svm(split_TRAIN_DATA,np.reshape(TRAIN_LABEL,(np.shape(TRAIN_LABEL)[0])))
    errors[i+10,1] = learner.test_svm(split_TEST_DATA,np.reshape(TEST_LABEL,(np.shape(TEST_LABEL)[0])))
    learner = SVM(kernel='sigmoid',C=2**(i))
    learner.train_svm(split_TRAIN_DATA,np.reshape(TRAIN_LABEL,(np.shape(TRAIN_LABEL)[0])))
    errors[i+10,2] = learner.test_svm(split_TEST_DATA,np.reshape(TEST_LABEL,(np.shape(TEST_LABEL)[0])))
    learner = SVM(kernel='poly',degree=2,C=2**(i))
    learner.train_svm(split_TRAIN_DATA,np.reshape(TRAIN_LABEL,(np.shape(TRAIN_LABEL)[0])))
    errors[i+10,3] = learner.test_svm(split_TEST_DATA,np.reshape(TEST_LABEL,(np.shape(TEST_LABEL)[0])))
    learner = SVM(kernel='poly',degree=3,C=2**(i))
    learner.train_svm(split_TRAIN_DATA,np.reshape(TRAIN_LABEL,(np.shape(TRAIN_LABEL)[0])))
    errors[i+10,4] = learner.test_svm(split_TEST_DATA,np.reshape(TEST_LABEL,(np.shape(TEST_LABEL)[0])))
    learner = SVM(kernel='poly',degree=4,C=2**(i))
    learner.train_svm(split_TRAIN_DATA,np.reshape(TRAIN_LABEL,(np.shape(TRAIN_LABEL)[0])))
    errors[i+10,5] = learner.test_svm(split_TEST_DATA,np.reshape(TEST_LABEL,(np.shape(TEST_LABEL)[0])))

np.save("svm_results",errors)

# import pylab as pl
# x = np.arange(-10,11,1)
# pl.plot(x,errors[:,0].T,label='Linear')
# pl.plot(x,errors[:,1].T,label='RBF')
# pl.plot(x,errors[:,2].T,label='Sigmoid')
# pl.plot(x,errors[:,3].T,label='Poly Deg 2')
# pl.plot(x,errors[:,4].T,label='Poly Deg 3')
# pl.plot(x,errors[:,5].T,label='Poly Deg 4')
# pl.legend(loc="upper left")
# pl.show()
