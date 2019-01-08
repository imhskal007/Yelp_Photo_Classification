import numpy as np
import pandas as pd
import h5py

# Paths
FEATURES_HOME = '/home/lux/Yelp_Classification/Code/features/'
DATA_HOME = '/home/lux/Yelp_Classification/Data/'

# Get the photo to business id mapping from the dataset
test_photo_to_business_ids = pd.read_csv(DATA_HOME + 'test_photo_to_biz.csv')

# Get unique business ids
business_ids = test_photo_to_business_ids.business_id.unique()
print('Total test business: ', len(business_ids))

# Reading stored image features from h5 file
test_features_file = h5py.File(FEATURES_HOME + 'test_features.h5')
#test_features = np.copy(test_features_file['feature'])
#test_features_file.close()

print test_features_file['feature'][0]

# Create a dataframe to store business id, labels and image features,
# set them up to feed into machine learning models
test_dataset = pd.DataFrame(columns=['business_id', 'feature'])

id = 0
for business_id in business_ids:
	#business_id = int(business_id)
	
	# 
	images_for_business_id = test_photo_to_business_ids[test_photo_to_business_ids['business_id'] == business_id].index.tolist()
	
	# 
	feature = list(np.mean(np.asarray(test_features_file['feature'][images_for_business_id[0]:(images_for_business_id[-1] + 1)]), axis = 0))

	# 
	test_dataset.loc[business_id] = [business_id, feature]
	id += 1
	if id % 100 == 0:
		print "ID:", id

print "test Business feature extraction is completed"
test_features_file.close()

with open(FEATURES_HOME + 'test_business_features.csv', 'w') as business_features_file:
	test_dataset.to_csv(business_features_file, index=False)