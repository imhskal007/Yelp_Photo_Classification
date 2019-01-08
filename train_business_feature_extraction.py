mport numpy as np
import pandas as pd
import h5py

# Paths
FEATURES_HOME = '/home/lux/Yelp_Classification/Code/features/'
DATA_HOME = '/home/lux/Yelp_Classification/Data/'

# Get the photo to business id mapping from the dataset
train_photo_to_business_ids = pd.read_csv(DATA_HOME + 'train_photo_to_biz_ids.csv')

# Getting the labels for each business
train_data_business_labels = pd.read_csv(DATA_HOME + 'train.csv').dropna()

# Sort labels in ascending order
train_data_business_labels['labels'] = train_data_business_labels['labels'].apply(lambda feature_vector: tuple(sorted(int(feature) for feature in feature_vector.split())))
train_data_business_labels.set_index('business_id', inplace = True)

# Get unique business ids
business_ids = train_data_business_labels.index.unique()
print('Total train business: ', len(business_ids))

# Reading stored image features from h5 file
train_features_file = h5py.File(FEATURES_HOME + 'train_features.h5')
train_features = np.copy(train_features_file['feature'])
train_features_file.close()

# Create a dataframe to store business id, labels and image features,
# set them up to feed into machine learning models
train_dataset = pd.DataFrame(columns=['business_id', 'label', 'feature'])

for business_id in business_ids:
	business_id = int(business_id)
	
	# Getting the labels for each business id
	label = train_data_business_labels.loc[business_id]['labels']
	
	# 
	images_for_business_id = train_photo_to_business_ids[train_photo_to_business_ids['business_id'] == business_id].index.tolist()
	
	# 
	feature = list(np.mean(train_features[images_for_business_id], axis = 0))

	# 
	train_dataset.loc[business_id] = [business_id, label, feature]

print('Train Business feature extraction is completed')

with open(FEATURES_HOME + 'train_business_features.csv', 'w') as business_features_file:
	train_dataset.to_csv(business_features_file, index=False)