#-------------------------------------------------------------------------------
# Project     | pysci-tutorial
# Module      | predicting minsit dataset tutorial
# Author      | sameh kamal
# Description | applying a set of learning models to minsit dataset
# Reference   | http://scikit-learn.org/stable/tutorial/basic/tutorial.html
#-------------------------------------------------------------------------------

from sklearn import datasets, svm, linear_model, metrics
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
expected                = minsit_test_set_target

# ------------------------------------------------------------------------------
# Support Vector Machine
# ------------------------------------------------------------------------------

# initialize support vector machine classifier
svm_classifier_obj = svm.SVC()

# train the svm model using the training set
svm_classifier_obj.fit( minsit_train_set, minsit_train_set_target )

# predict the testing set using svm model
predicted_svm = svm_classifier_obj.predict( minsit_test_set )

# print svm classifier parameters
print("------------------------------------------------------------------\n")
print("classifier: %s \n" % svm_classifier_obj)
print("------------------------------------------------------------------\n")
# print classification model report
print("%s" % metrics.classification_report(expected, predicted_svm) )
print("------------------------------------------------------------------\n")

# ------------------------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------------------------

# initialize logistic regression classifier object
logreg_classifer_obj = linear_model.LogisticRegression()

# train the logistic regression classifier on minsit training set
logreg_classifer_obj.fit( minsit_train_set,minsit_train_set_target )

# predict the testing set using logistic regression model
predicted_logreg = logreg_classifer_obj.predict( minsit_test_set )

# print logestic regression classifier parameters
print("------------------------------------------------------------------\n")
print("classifier: %s \n" % logreg_classifer_obj)
print("------------------------------------------------------------------\n")
# print classification model report
print("%s" % metrics.classification_report(expected, predicted_logreg) )
print("------------------------------------------------------------------\n")
