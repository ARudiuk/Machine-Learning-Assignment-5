import scipy.io
data = scipy.io.loadmat('NewsGroup.mat')
TRAIN_LABEL = data['TRAIN_LABEL']
TEST_LABEL = data['TEST_LABEL']

import numpy as np
import math as math
#constants from info file
prob_class_1 = 0.574186827179
prob_class_2 = 0.425813172821

prob_word = np.loadtxt('word_probability.txt')
split_TEST_DATA = np.load("split_TEST_DATA.npy")
split_TRAIN_DATA = np.load("split_TRAIN_DATA.npy")

#calculate accuracy on train set
train_correct = 0
for i in range(np.shape(split_TRAIN_DATA)[0]):
    prob_given_x_c1 = 1
    prob_given_x_c2 = 1
    prob_given_x_c1 += math.log(prob_class_1)
    prob_given_x_c2 += math.log(prob_class_2)
    for j in range(np.shape(split_TRAIN_DATA)[1]):
        if split_TRAIN_DATA[i,j]!=0:
            prob_given_x_c1 += math.log(prob_word[j,0])
            prob_given_x_c2 += math.log(prob_word[j,1])
    if prob_given_x_c1>prob_given_x_c2:
        if TRAIN_LABEL[i,0] == 1:
            train_correct+=1
    else:
        if TRAIN_LABEL[i,0] == 2:
            train_correct+=1
print "Percentage correct was", float(train_correct)/float(np.shape(TRAIN_LABEL)[0])

test_correct = 0
for i in range(np.shape(split_TEST_DATA)[0]):
    prob_given_x_c1 = 1
    prob_given_x_c2 = 1
    prob_given_x_c1 += math.log(prob_class_1)
    prob_given_x_c2 += math.log(prob_class_2)
    for j in range(np.shape(split_TEST_DATA)[1]):
        if split_TEST_DATA[i,j]!=0:
            prob_given_x_c1 += math.log(prob_word[j,0])
            prob_given_x_c2 += math.log(prob_word[j,1])
    if prob_given_x_c1>prob_given_x_c2:
        if TEST_LABEL[i,0] == 1:
            test_correct+=1
    else:
        if TEST_LABEL[i,0] == 2:
            test_correct+=1
print "Percentage correct was", float(test_correct)/float(np.shape(TEST_LABEL)[0])
