import numpy as np
errors = np.load("svm_results.npy")
import pylab as pl
x = np.arange(-5,11,1)
y1 = errors[:,0].T
y2 = errors[:,1].T
y3 = errors[:,2].T
y4 = errors[:,3].T
y5 = errors[:,4].T
y6 = errors[:,5].T
errors = np.load("svm_results_2.npy")
y1 = np.append(y1,errors[2:,0].T)
y2 = np.append(y2,errors[2:,1].T)
y4 = np.append(y4,errors[2:,3].T)
errors = np.load("svm_results_train.npy")
x2 = np.arange(-5,6,1)
y7 = errors[:,0].T
y8 = errors[:,1].T
y9 = errors[:,2].T
y10 = errors[:,3].T
y11 = errors[:,4].T
y12 = errors[:,5].T
pl.plot(x,y1,label='Linear')
pl.plot(x,y2,label='RBF')
pl.plot(x2,y3,label='Sigmoid')
pl.plot(x,y4,label='Poly Deg 2')
pl.plot(x2,y5,label='Poly Deg 3')
pl.plot(x2,y6,label='Poly Deg 4')

pl.plot(x2,y7,label='Linear Train')
pl.plot(x2,y8,label='RBF Train')
pl.plot(x2,y9,label='Sigmoid Train')
pl.plot(x2,y10,label='Poly Deg 2 Train')
pl.plot(x2,y11,label='Poly Deg 3 Train')
pl.plot(x2,y12,label='Poly Deg 4 Train')
pl.legend(loc="center left")
pl.xlabel("2^x = C")
pl.ylabel("Percentage Correct")
pl.title("Accuracy vs C value for SVM")
pl.show()
