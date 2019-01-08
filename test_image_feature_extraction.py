# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 12:14:31 2018
@author: Vinothini Pushparaja & Lakshmi Vaidiyanathan
"""

import numpy as np
import pandas as pd
import h5py
import os
import time

from resnet152 import ResNet152
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model

# Paths
FEATURES_HOME = '/home/lux/Yelp_Classification/Code/features/'
DATA_HOME = '/home/lux/Yelp_Classification/Data/'

model = ResNet152(include_top=False, weights='imagenet')

def extractFeatures(image_paths):
	test_size = len(image_paths)
	img_array = np.zeros((test_size, 224, 224, 3))

	for i, image_path in enumerate(image_paths):
		img = image.load_img(image_path, target_size = (224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis = 0)
		x = preprocess_input(x)
		img_array[i] = x
		
	features = model.predict(img_array)
	features = features.reshape((test_size, 2048))
	return features

if not os.path.isfile(FEATURES_HOME + 'test_features.h5'):
    file = h5py.File(FEATURES_HOME + 'test_features.h5', 'w')
    photoId = file.create_dataset('photoId', (0,), maxshape = (None,), dtype = '|S54')
    feature = file.create_dataset('feature', (0, 4096), maxshape = (None, 4096))
    file.close()
    
file = h5py.File(FEATURES_HOME + 'test_features.h5', 'r+')
already_extracted_images = len(file['photoId'])
file.close()

test_set = pd.read_csv(DATA_HOME + 'test_photo_to_biz.csv')
test_image_paths = [os.path.join(DATA_HOME + 'test_photos/', str(photo_id) + '.jpg') for photo_id in test_set['photo_id']]

test_size = len(test_image_paths)
batch_size = 500
batch_number = already_extracted_images / batch_size + 1

print('Total images: ' + str(test_size))
print('Already done images: ' + str(already_extracted_images))

for image_count in range(already_extracted_images, test_size, batch_size):
    start_time = time.time()
    image_paths = test_image_paths[image_count: min(image_count + batch_size, test_size)]
    
    features = extractFeatures(image_paths)
    
    total_done_images = image_count + features.shape[0]
    
    file = h5py.File(FEATURES_HOME + 'test_features.h5', 'r+')
    file['photoId'].resize((total_done_images,))
    file['photoId'][image_count: total_done_images] = np.array(image_paths)
    file['feature'].resize((total_done_images, features.shape[1]))
    file['feature'][image_count: total_done_images, :] = features
    file.close()
    
    
    print("Batch No:", batch_number, "\tStart:", image_count, "\tEnd:", image_count + batch_size, "\tTime required:", time.time() - start_time, "sec", "\tCompleted:", float(image_count + batch_size) / float(test_size) * 100, "%")
    batch_number += 1