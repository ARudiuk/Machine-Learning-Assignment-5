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
#get number of words and documents
import numpy as np
word_count = 0
document_count = 0
for i in range(np.shape(TRAIN_DATA)[0]):
    if TRAIN_DATA[i][1]>word_count:
        word_count = TRAIN_DATA[i][1]
    if TRAIN_DATA[i][0]>document_count:
        document_count=TRAIN_DATA[i][0]
#get number of words in each class
word_count_class_1 = 0
word_count_class_2 = 0
for i in range(np.shape(TRAIN_DATA)[0]):
    #-1 because of index starting at 0 not 1 like documen id does
    if TRAIN_LABEL[TRAIN_DATA[i][0]-1][0] == 1:
        word_count_class_1+=TRAIN_DATA[i][2]
    else:
        word_count_class_2+=TRAIN_DATA[i][2]
print "There are",document_count,"unique documents"
print "There are",word_count,"unique words"
print "There are",word_count_class_1,"words in the first class"
print "There are",word_count_class_2,"words in the second class"
#split hte data into documents
split_TRAIN_DATA = np.zeros((document_count,word_count))
current_document = TRAIN_DATA[0,0]
for i in range(np.shape(TRAIN_DATA)[0]):
    split_TRAIN_DATA[TRAIN_DATA[i][0]-1,TRAIN_DATA[i][1]-1]=TRAIN_DATA[i][2]
print "The split train data is", np.shape(split_TRAIN_DATA)

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
    prob_word[i,0]=float(word_sum_class_1)/float(word_count_class_1)
    prob_word[i,1]=float(word_sum_class_2)/float(word_count_class_2)
np.savetxt('word_probability.txt',prob_word)

#probabity of words times class
#Test data processing
# word_count = 0
# document_count = 0
# for i in range(np.shape(TEST_DATA)[0]):
#     if TEST_DATA[i][1]>word_count:
#         word_count = TEST_DATA[i][1]
#     if TEST_DATA[i][0]>document_count:
#         document_count=TEST_DATA[i][0]
# print "There are",document_count,"unique documents"
# print "There are",word_count,"unique words"
# split_TEST_DATA = np.zeros((document_count,word_count))
# current_document = TEST_DATA[0,0]
# for i in range(np.shape(TEST_DATA)[0]):
#     split_TEST_DATA[TEST_DATA[i][0]-1,TEST_DATA[i][1]-1]=TEST_DATA[i][2]
# print "The split TEST data is", np.shape(split_TEST_DATA)
# print split_TEST_DATA[0,:]