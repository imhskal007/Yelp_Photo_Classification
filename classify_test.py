import pandas as pd
import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import MultiLabelBinarizer

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

# Read the test data set
test_data = pd.read_csv(FEATURES_HOME + 'test_business_features.csv')

# 
testX = np.array([get_features(feature) for feature in test_data['feature']])

mlb = MultiLabelBinarizer()
##########################
#Support Vector Machine  #
##########################

svm = joblib.load(MODELS_HOME + 'svm_model_for_validation.pkl')

print 'Predicting SVM Test data'
svm_pred = svm.predict(testX)

svm_pred_labels = mlb.inverse_transform(svm_pred)
svm_pred_labels[0]

test_data['labels'] = svm_pred_labels
test_data['labels'] = test_data.labels.str.replace(',', '').replace('(', '').replace(')', '')
#test_data['labels'] = test_data.labels.str.replace('(', '')
#test_data['labels'] = test_data.labels.str.replace(')', '')
submit_1 = test_data[['business_id', 'labels']]
submit_1.to_csv('submit_sv.csv', index=None)

print 'Predicting SVM done'
##########################
#Random Forest           #
##########################

rf = joblib.load(MODELS_HOME + 'rfc_model_for_validation.pkl')

print 'Predicting Random Forest Test data'
rf_pred = rf.predict(testX)

rf_pred_labels = mlb.inverse_transform(rf_pred)

test_data['labels_rf'] = rf_pred_labels
test_data['labels_rf'] = test_data.labels.str.replace(',', '').replace('(', '').replace(')', '')
#test_data['labels'] = test_data.labels.str.replace('(', '')
#test_data['labels'] = test_data.labels.str.replace(')', '')
submit_1 = test_data[['business_id', 'labels_rf']]
submit_1.to_csv('submit_rf.csv', index=None)

print 'Predicting RF done'