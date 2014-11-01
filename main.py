#looking for p(y|x)
#inverse is p(x|y) known as generative model
#p(y|x) = p(x|y)*p(y)/p(x)
#p(x(i)) probability of word at index i
#p(x(i)|y;theta) = bernoulli random variable. Binomial distribution = P(i,1)^y*(1-P(i,1))^1-y

#import data
import scipy.io
data = scipy.io.loadmat('NewsGroup.mat')
TRAIN_DATA = data['TRAIN_DATA']
TRAIN_LABEL = data['TRAIN_LABEL']
print TRAIN_DATA.shape
print TRAIN_LABEL.shape
TEST_DATA = data['TEST_DATA']
TEST_LABEL = data['TEST_LABEL']
print TEST_DATA.shape   
print TEST_LABEL.shape
#get number of words
import numpy as np
word_count = 0
for i in range(np.shape(TRAIN_DATA)[0]):
    if TRAIN_DATA[i][1]>word_count:
        word_count = TRAIN_DATA[i][1]
for i in range(np.shape(TEST_DATA)[0]):
    if TEST_DATA[i][1]>word_count:
        word_count = TEST_DATA[i][1]
#get number of words in each class
word_count_class_1 = 0
word_count_class_2 = 0
for i in range(np.shape(TRAIN_DATA)[0]):
    #-1 because of index starting at 0 not 1 like documen id does
    if TRAIN_LABEL[TRAIN_DATA[i][0]-1][0] == 1:
        word_count_class_1+=TRAIN_DATA[i][2]
    else:
        word_count_class_2+=TRAIN_DATA[i][2]
print "There are",np.shape(TRAIN_LABEL)[0],"unique documents"
print "There are",word_count,"unique words"
print "There are",word_count_class_1,"words in the first class"
print "There are",word_count_class_2,"words in the second class"
#split the data into documents
split_TRAIN_DATA = np.zeros((np.shape(TRAIN_LABEL)[0],word_count))
for i in range(np.shape(TRAIN_DATA)[0]):
    split_TRAIN_DATA[TRAIN_DATA[i][0]-1,TRAIN_DATA[i][1]-1]=TRAIN_DATA[i][2]
print "The split train data shape is", np.shape(split_TRAIN_DATA)

split_TEST_DATA = np.zeros((np.shape(TEST_LABEL)[0],word_count))
for i in range(np.shape(TEST_DATA)[0]):
    split_TEST_DATA[TEST_DATA[i][0]-1,TEST_DATA[i][1]-1]=TEST_DATA[i][2]
print "The split TEST data shape is", np.shape(split_TEST_DATA)


prob_word = np.zeros((word_count,2))

#sum number of occurances of word in each class and calculate value over all words in the class
for i in range(word_count):
    word_sum_class_1 = 0
    word_sum_class_2 = 0
    for j in range(np.shape(split_TRAIN_DATA)[0]):
        if(TRAIN_LABEL[j,0]==1):
            word_sum_class_1+=split_TRAIN_DATA[j,i]
        else:
            word_sum_class_2+=split_TRAIN_DATA[j,i]
    prob_word[i,0]=float(word_sum_class_1+.1)/float(word_count_class_1)
    prob_word[i,1]=float(word_sum_class_2+.1)/float(word_count_class_2)
np.savetxt('word_probability.txt',prob_word)
prob_word = np.loadtxt('word_probability.txt')
print "Shape of P(x|C) matrix is",np.shape(prob_word)
prob_class_1 = float(word_count_class_1)/float(word_count_class_1+word_count_class_2)
print prob_class_1
prob_class_2 = 1.0-prob_class_1
print prob_class_2

#calculate accuracy on train set
train_correct = 0
for i in range(np.shape(split_TRAIN_DATA)[0]):
    prob_given_x_c1 = 1
    prob_given_x_c2 = 1
    for j in range(np.shape(split_TRAIN_DATA)[1]):
        if split_TRAIN_DATA[i,j]!=0:
            prob_given_x_c1 = prob_class_1*prob_word[j,0]
            prob_given_x_c2 = prob_class_2*prob_word[j,1]
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
    for j in range(np.shape(split_TEST_DATA)[1]):
        if split_TEST_DATA[i,j]!=0:
            prob_given_x_c1 = prob_class_1*prob_word[j,0]
            prob_given_x_c2 = prob_class_2*prob_word[j,1]
    if prob_given_x_c1>prob_given_x_c2:
        if TEST_LABEL[i,0] == 1:
            test_correct+=1
    else:
        if TEST_LABEL[i,0] == 2:
            test_correct+=1
print "Percentage correct was", float(test_correct)/float(np.shape(TEST_LABEL)[0])


#probabity of words times class