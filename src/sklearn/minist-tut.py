#-------------------------------------------------------------------------------
# Project     | pysci-tutorial
# Module      | predicting minsit dataset tutorial
# Author      | sameh kamal
# Description | applying a set of learning models to minsit dataset
# Reference   | http://scikit-learn.org/stable/tutorial/basic/tutorial.html
#-------------------------------------------------------------------------------

from sklearn import datasets, svm
from numpy   import array

# load minsit dataset
minsit = datasets.load_digits()
dt_count  = len(minsit.images)
data      = minsit.images.reshape((dt_count, -1))

# define the training set
minsit_train_set        = data[ 1 : dt_count/2 ]
minsit_train_set_target = minsit.target[ 1 : dt_count/2 ]

# define the testing set
minsit_test_set         = data[ dt_count/2 : -1]
minsit_test_set_target  = minsit.target[ dt_count/2 : -1 ]

# Create a classifier: a support vector machine classifier
svm_model_obj = svm.SVC(gamma=0.001)

# train the svm model using the training set
svm_model_obj.fit( minsit_train_set, minsit_train_set_target )

# predict the testing set
expected  = minsit_test_set_target
predicted = svm_model_obj.predict( minsit_test_set )

# calculate errors array (1 for error 0 otherwise)
errors    = array([ [1,0][x==y] for x, y in zip(expected, predicted)])

# print model accuracy 1 - error percentage
print("# accuracy: %2.3f %%" % ( (1 - sum(errors)/len(errors)) * 100 ) )
