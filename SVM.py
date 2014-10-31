from sklearn import svm
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
