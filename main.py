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

import numpy as np
word_count = 0
document_count = 0
for i in range(np.shape(TRAIN_DATA)[0]):
    if TRAIN_DATA[i][1]>word_count:
        word_count = TRAIN_DATA[i][1]
    if TRAIN_DATA[i][0]>document_count:
        document_count=TRAIN_DATA[i][0]
print "There are",document_count,"unique documents"
print "There are",word_count,"unique words"
split_TRAIN_DATA = np.zeros((document_count,word_count))
current_document = TRAIN_DATA[0,0]
for i in range(np.shape(TRAIN_DATA)[0]):
    split_TRAIN_DATA[TRAIN_DATA[i][0]-1,TRAIN_DATA[i][1]-1]=TRAIN_DATA[i][2]
print "The split train data is", np.shape(split_TRAIN_DATA)
print split_TRAIN_DATA[0,:]

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