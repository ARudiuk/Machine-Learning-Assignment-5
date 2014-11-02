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
TEST_DATA = data['TEST_DATA']
TEST_LABEL = data['TEST_LABEL']
print "data imported"
#get number of words
import numpy as np
word_count = 0
for i in range(np.shape(TRAIN_DATA)[0]):
    if TRAIN_DATA[i][1]>word_count:
        word_count = TRAIN_DATA[i][1]
word_count_train = word_count
for i in range(np.shape(TEST_DATA)[0]):
    if TEST_DATA[i][1]>word_count:
        word_count = TEST_DATA[i][1]
print "words counted"
#split the data into documents
split_TRAIN_DATA = np.zeros((np.shape(TRAIN_LABEL)[0],word_count))
for i in range(np.shape(TRAIN_DATA)[0]):
    split_TRAIN_DATA[TRAIN_DATA[i][0]-1,TRAIN_DATA[i][1]-1]=TRAIN_DATA[i][2]
split_TEST_DATA = np.zeros((np.shape(TEST_LABEL)[0],word_count))
for i in range(np.shape(TEST_DATA)[0]):
    split_TEST_DATA[TEST_DATA[i][0]-1,TEST_DATA[i][1]-1]=TEST_DATA[i][2]
np.save("split_TRAIN_DATA",split_TRAIN_DATA)
np.save("split_TEST_DATA",split_TEST_DATA)
print "data split into documents"

#get number of words in each class
word_count_class_1 = 0
word_count_class_2 = 0
for i in range(np.shape(TRAIN_DATA)[0]):
    #-1 because of index starting at 0 not 1 like documen id does
    if TRAIN_LABEL[TRAIN_DATA[i][0]-1][0] == 1:
        word_count_class_1+=TRAIN_DATA[i][2]
    else:
        word_count_class_2+=TRAIN_DATA[i][2]
word_count_class_1_test = 0
word_count_class_2_test = 0
for i in range(np.shape(TEST_DATA)[0]):
    #-1 because of index starting at 0 not 1 like documen id does
    if TEST_LABEL[TEST_DATA[i][0]-1][0] == 1:
        word_count_class_1_test+=TEST_DATA[i][2]
    else:
        word_count_class_2_test+=TEST_DATA[i][2]
print "words counted for each class"

#sum number of occurances of word in each class and calculate the value over all words in the class
prob_word = np.zeros((word_count,2))
for i in range(word_count):
    if(i%5000==0):
        print i,"word probabilities calculated"
    word_sum_class_1 = 0
    word_sum_class_2 = 0
    if i<word_count_train:
        for j in range(np.shape(split_TRAIN_DATA)[0]):
            if(TRAIN_LABEL[j,0]==1):
                word_sum_class_1+=split_TRAIN_DATA[j,i]
            else:
                word_sum_class_2+=split_TRAIN_DATA[j,i]
    prob_word[i,0]=float(word_sum_class_1+1)/float(word_count_class_1+word_count)
    prob_word[i,1]=float(word_sum_class_2+1)/float(word_count_class_2+word_count)
np.savetxt('word_probability.txt',prob_word)
print "probability of each word counted"

#print info to console and file
print "There are",np.shape(TRAIN_LABEL)[0],"unique documents in the train data"
print "There are",np.shape(TEST_LABEL)[0],"unique documents in the test data"
print "There are",word_count_train,"unique words in the train data"
print "There are",word_count,"unique words in the test data"
print "There are",word_count_class_1,"words in the first class for the train data"
print "There are",word_count_class_2,"words in the second class for the train data"
print "There are",word_count_class_1_test,"words in the first class for the test data"
print "There are",word_count_class_2_test,"words in the second class for the test data"
print "Shape of P(x|C) matrix is",np.shape(prob_word)
prob_class_1 = float(word_count_class_1)/float(word_count_class_1+word_count_class_2)
print "The probability of a word being class one is",prob_class_1,"in the training data set"
prob_class_2 = 1.0-prob_class_1
print "The probability of a word being class two is", prob_class_2,"in the training data set"
prob_class_1 = float(word_count_class_1_test)/float(word_count_class_1_test+word_count_class_2_test)
print "The probability of a word being class one is",prob_class_1,"in the test data set"
prob_class_2 = 1.0-prob_class_1
print "The probability of a word being class two is", prob_class_2,"in the test data set"


f = open("info.txt",'w')
f.write("There are "+str(np.shape(TRAIN_LABEL)[0])+" unique documents in the train data\n")
f.write("There are "+str(np.shape(TEST_LABEL)[0])+" unique documents in the test data\n")
f.write("There are "+str(word_count_train)+" unique words in the train data\n")
f.write("There are "+str(word_count)+" unique words in the test data\n")
f.write("There are "+str(word_count_class_1)+" words in the first class for the train data\n")
f.write("There are "+str(word_count_class_2)+" words in the second class for the train data\n")
f.write("There are "+str(word_count_class_1_test)+" words in the first class for the test data\n")
f.write("There are "+str(word_count_class_2_test)+" words in the second class for the test data\n")
f.write("Shape of P(x|C) matrix is "+str(np.shape(prob_word))+" \n")
prob_class_1 = float(word_count_class_1)/float(word_count_class_1+word_count_class_2)
f.write("The probability of a word being class one is "+str(prob_class_1)+" in the training data set\n")
prob_class_2 = 1.0-prob_class_1
f.write("The probability of a word being class two is "+str(prob_class_2)+" in the training data set\n")
prob_class_1 = float(word_count_class_1_test)/float(word_count_class_1_test+word_count_class_2_test)
f.write("The probability of a word being class one is "+str(prob_class_1)+" in the test data set\n")
prob_class_2 = 1.0-prob_class_1
f.write("The probability of a word being class two is " + str(prob_class_2)+" in the test data set\n")
f.close()
