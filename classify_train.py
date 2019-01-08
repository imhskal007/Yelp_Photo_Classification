# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 1.02.31 2018
@author: Lakshmi Vaidiyanathan & Vinothini Pushparaja 
"""
import os
import pandas as pd
import numpy as np
import time
import pickle

from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC

def get_labels(label_string):
    """
        This function converts label from string to array of labels
        Input: "(1, 2, 3, 4, 5)"
        Output: [1, 2, 3, 4, 5]
    """
    label_array = label_string[1:-1]
    label_array = label_array.split(',')
    label_array = [int(label) for label in label_array if len(label) > 0]
    return label_array


def get_features(feature_string):
    """
        This function converts feature vector from string to array of features
        Input: "(1.2, 3.4, ..., 9.10)"
        Output: [1.2, 3.4, ..., 9.10]
    """
    feature_array = feature_string[1:-1]
    feature_array = feature_array.split(',')
    feature_array = [float(label) for label in feature_array]
    return feature_array


# Set home paths for data and features
# Paths
FEATURES_HOME = '/home/lux/Yelp_Classification/Code/features/'
DATA_HOME = '/home/lux/Yelp_Classification/Data/'
MODELS_HOME = '/home/lux/Yelp_Classification/Code/train_model/'

# Read training data and test data
train_data = pd.read_csv(FEATURES_HOME + 'train_business_features.csv')

# Separate the labels from features in the training data
X = np.array([get_features(feature) for feature in train_data['feature']])
y = np.array([get_labels(label) for label in train_data['label']])

# Binary representation (just like one-hot vector) (1, 3, 5, 9) -> (1, 0, 1, 0, 1, 0, 0, 0, 1)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)

random_state = np.random.RandomState(2018)

#################################
# Support Vector Machine        #
#################################

print "Support Vector Machine Started"
if not os.path.isfile(MODELS_HOME + 'svm_model_for_validation.pkl'):
    print "Model training started."

    # Start time
    start_time = time.time()

    # Create an SVM classifier from sklearn package
    clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, verbose=True))

    # Fit the classifier on the training data and labels
    clf.fit(X, y)

    print "Model trained."

    joblib.dump(clf, MODELS_HOME + 'svm_model_for_validation.pkl')
    print "Model saved."

    # End time
    end_time = time.time()

    print "Time taken for training the SVM model:", end_time - start_time, "sec"

clf = joblib.load(MODELS_HOME + 'svm_model_for_validation.pkl')

print "Model loaded."

# Predict the labels for the validation data
svm_preds_binary = clf.predict(X)

# Predicted labels are converted back
# (1, 0, 1, 0, 1, 0, 0, 0, 1) -> (1, 3, 5, 9)
predicted_labels = mlb.inverse_transform(svm_preds_binary)

print "Validation Set Results:"
print "Overall F1 Score:", f1_score(svm_preds_binary, y, average='micro')
print "Individual F1 Score:", f1_score(svm_preds_binary, y, average=None)
print "\n"
print "Overall Precision Score:", precision_score(svm_preds_binary, y, average='micro')
print "Individual Precision Score:", precision_score(svm_preds_binary, y, average=None)
print "\n"
print "Overall Recall Score:", recall_score(svm_preds_binary, y, average='micro')
print "Individual Recall Score:", recall_score(svm_preds_binary, y, average=None)

print "\n Support Vector Machine Ends"

#################################
# Random Forest			#
#################################
print "\nRandom Forest Modeling starts"

if not os.path.isfile(MODELS_HOME + 'rfc_model_for_validation.pkl'):
	print "Random Forest Model training started."

	# Start time
	start_time_rf = time.time()
	rfc = RandomForestClassifier(
		n_jobs=-1,
		n_estimators=1000,
		min_samples_split=2,
		random_state=random_state,
		verbose=0)
	ovr_rfc = OneVsRestClassifier(estimator=rfc, n_jobs=1)

	# Fit the classifier on the training data and labels
	ovr_rfc.fit(X, y)
	print "Random Forest Model trained."
	joblib.dump(ovr_rfc, MODELS_HOME + 'rfc_model_for_validation.pkl')
	print "Random Forest Model saved."

	# End time
	end_time_rf = time.time()
	print "Time taken for training the Random Forest model:", end_time_rf - start_time_rf, "sec"

rfc = joblib.load(MODELS_HOME + 'rfc_model_for_validation.pkl')
print "RF Model loaded"

# Predict the labels for the validation data
rfc_preds_binary = rfc.predict(X)

# Predicted labels are converted back
# (1, 0, 1, 0, 1, 0, 0, 0, 1) -> (1, 3, 5, 9)
rfc_predicted_labels = mlb.inverse_transform(rfc_preds_binary)

print "Validation RF Set Results:"
print "Overall F1 Score:", f1_score(rfc_preds_binary, y, average='micro')
print "Individual F1 Score:", f1_score(rfc_preds_binary, y, average=None)
print "\n"
print "Overall Precision Score:", precision_score(rfc_preds_binary, y, average='micro')
print "Individual Precision Score:", precision_score(rfc_preds_binary, y, average=None)
print "\n"
print "Overall Recall Score:", recall_score(rfc_preds_binary, y, average='micro')
print "Individual Recall Score:", recall_score(rfc_preds_binary, y, average=None)

print "\n Random Forest Ends"

print '*********************************'

#################################
# Logistic Regression           #
#################################

print "Logistic Regression Model starts\n"

if not os.path.isfile(MODELS_HOME + 'logreg_model_for_validation.pkl'):
    print("Model training started.")

    # Start time
    start_time = time.time()

    # Create an Logistic Regression classifier from sklearn package
    logreg = OneVsRestClassifier(LogisticRegression(random_state = 0))

    # Fit the classifier on the training data and labels
    logreg.fit(X, y)

    print("Model trained.")

    joblib.dump(clf, MODELS_HOME + 'logreg_model_for_validation.pkl',compress=9)
    print("Model saved.")

    # End time
    end_time = time.time()

    print("Time taken for training the Logistic Regression model:", end_time - start_time, "sec")

logreg = joblib.load(MODELS_HOME + 'logreg_model_for_validation.pkl')

print("Model loaded.")

# Predict the labels for the validation data
logreg_preds_binary = clf.predict(X)
print('logreg_preds_binary: ', logreg_preds_binary)

# Predicted labels are converted back
# (1, 0, 1, 0, 1, 0, 0, 0, 1) -> (1, 3, 5, 9)
predicted_labels = mlb.inverse_transform(logreg_preds_binary)
print('predicted_labels: ', predicted_labels)

print("Validation Set Results:")
print("Overall F1 Score:", f1_score(logreg_preds_binary, y, average='micro'))
print("Individual F1 Score:", f1_score(logreg_preds_binary, y, average=None))
print "\n"
print "Overall Precision Score:", precision_score(logreg_preds_binary, y, average='micro')
print "Individual Precision Score:", precision_score(logreg_preds_binary, y, average=None)
print "\n"
print "Overall Recall Score:", recall_score(logreg_preds_binary, y, average='micro')
print "Individual Recall Score:", recall_score(logreg_preds_binary, y, average=None)

print "Logistic Regression Model Ends"

print '*********************************'
print '*********************************'


# Read the test data set
test_data = pd.read_csv(FEATURES_HOME + 'test_business_features.csv')

#
testX = np.array([get_features(feature) for feature in test_data['feature']])

##########################
#Support Vector Machine  #
##########################

print 'Predicting SVM Test data'
svm_pred = clf.predict(testX)

svm_pred_labels = mlb.inverse_transform(svm_pred)
svm_pred_labels[0]

test_data['labels'] = svm_pred_labels
test_data['labels'] = test_data['labels'].str.replace(',', '').replace('(', '').replace(')', '')
#test_data['labels'] = test_data.labels.str.replace('(', '')
#test_data['labels'] = test_data.labels.str.replace(')', '')
submit_1 = test_data[['business_id', 'labels']]
submit_1.to_csv('submit_sv.csv', index=None)

print 'Predicting SVM done'

##########################
#Random Forest           #
##########################

print 'Predicting Random Forest Test data'
rf_pred = rfc.predict(testX)

rf_pred_labels = mlb.inverse_transform(rf_pred)

test_data['labels_rf'] = rf_pred_labels
test_data['labels_rf'] = test_data['labels_rf'].str.replace(',', '').replace('(', '').replace(')', '')
#test_data['labels'] = test_data.labels.str.replace('(', '')
#test_data['labels'] = test_data.labels.str.replace(')', '')
submit_1 = test_data[['business_id', 'labels_rf']]
submit_1.to_csv('submit_rf.csv', index=None)

print 'Predicting RF done'