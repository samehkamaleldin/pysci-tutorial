#-------------------------------------------------------------------------------
# Project     | pysci-tutorial
# Module      | minsit-tut
# Author      | sameh kamal
# Description | minsit dataset tutorial
# Reference   | http://scikit-learn.org/stable/tutorial/basic/tutorial.html
#-------------------------------------------------------------------------------

from sklearn import datasets, svm, linear_model, metrics
from numpy   import array

# ------------------------------------------------------------------------------
# MINSIT Dataset
# ------------------------------------------------------------------------------
# load minsit dataset
minsit = datasets.load_digits()
dt_count  = len(minsit.images)
data      = minsit.images.reshape((dt_count, -1))
# define the training set
minsit_train_set        = data[ 1 : dt_count/2 ]
minsit_train_set_target = minsit.target[ 1 : dt_count/2 ]
# define the testing set
minsit_test_set         = data[ dt_count/2 : -1]
minsit_test_set_lenght  = len(minsit_test_set)
minsit_test_set_target  = minsit.target[ dt_count/2 : -1 ]
expected                = minsit_test_set_target
# ------------------------------------------------------------------------------
# Support Vector Machine
# ------------------------------------------------------------------------------
# initialize support vector machine classifier
svm_classifier_obj = svm.SVC(gamma=0.001)
# train the svm model using the training set
svm_classifier_obj.fit( minsit_train_set, minsit_train_set_target )
# predict the testing set using svm model
predicted_svm = svm_classifier_obj.predict( minsit_test_set )
svm_err_count = sum(array([ [1,0][x==y] for x, y in zip(expected, predicted_svm)]))
svm_tp_count  = minsit_test_set_lenght - svm_err_count
# ------------------------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------------------------
# initialize logistic regression classifier object
logreg_classifer_obj = linear_model.LogisticRegression()
# train the logistic regression classifier on minsit training set
logreg_classifer_obj.fit( minsit_train_set,minsit_train_set_target )
# predict the testing set using logistic regression model
predicted_logreg = logreg_classifer_obj.predict( minsit_test_set )
logreg_err_count = sum(array([ [1,0][x==y] for x, y in zip(expected, predicted_logreg)]))
logreg_tp_count  = minsit_test_set_lenght - logreg_err_count
# print model test results
print("------------------------------------------------------------------")
print(" TEST SAMPLE SIZE      : %4.0f" % minsit_test_set_lenght )
print("------------------------------------------------------------------")
print(" SVM    TRUE. PREDICT  : %4.0f  -  %2.2f %%" % ( svm_tp_count   , svm_tp_count / minsit_test_set_lenght    * 100) )
print(" LOGREG TRUE. PREDICT  : %4.0f  -  %2.2f %%" % ( logreg_tp_count, logreg_tp_count / minsit_test_set_lenght * 100) )
print("------------------------------------------------------------------")
