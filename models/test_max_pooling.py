import pandas as pd 
import numpy as np 
import pdb

import multi_triplet_siamese_cnn 
#import triplet_siamese_cnn 

import plain_cnn 


univariate_random_20 = ['Lighting2', 'uWaveGestureLibrary_X', 'MoteStrain',
       'uWaveGestureLibrary_Z', 'ECG200', 'ProximalPhalanxOutlineAgeGroup',
       'TwoLeadECG', 'Coffee', 'Cricket_Y', 'SonyAIBORobotSurface',
       'NonInvasiveFatalECG_Thorax2', 'FacesUCR', 'MiddlePhalanxTW',
       'Yoga', 'Beef', 'FacesUCR', 'DistalPhalanxOutlineAgeGroup',
       'ToeSegmentation1', 'Two_Patterns', 'DistalPhalanxOutlineAgeGroup']


# Create dataframe for max-pooling
df = []
max_pool_vals = [.05, .1,.15, .2, .25, .3, .35, .5, .45, .5, .75, 1.0]
for pool_val in max_pool_vals:
	# Only perform one of these across all max pooling values 
	for dataset in univariate_random_20:
		print 'DATASET: ', dataset
		print 'POOL_PCTG: ', pool_val, '\n'
		acc = plain_cnn.test_model(dataset, pool_pctg=pool_val)
		df.append({'network': 'triplet_siamese_cnn', 'pool_size': pool_val, 'dataset': dataset, 'accuracy': acc})

		dataframe = pd.DataFrame(df)
		dataframe.to_csv('pooling_size_comparison.csv', index=False)

