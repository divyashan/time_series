import pandas as pd 
import numpy as np 
import pdb

import multi_triplet_siamese_cnn 
#import triplet_siamese_cnn 

import plain_cnn 

done_8 = ['Lighting2', 'uWaveGestureLibrary_X', 'MoteStrain','ECG200', 'ProximalPhalanxOutlineAgeGroup',
       'TwoLeadECG', 'Coffee', 'Cricket_Y', 'SonyAIBORobotSurface', 'uWaveGestureLibrary_Z',
       'NonInvasiveFatalECG_Thorax2', 'FacesUCR', 'MiddlePhalanxTW',
       'Yoga', 'Beef', 'FacesUCR', 'DistalPhalanxOutlineAgeGroup',
       'ToeSegmentation1']

univariate_random_20 = ['Lighting2', 'uWaveGestureLibrary_X', 'MoteStrain',
       'uWaveGestureLibrary_Z', 'ECG200', 'ProximalPhalanxOutlineAgeGroup',
       'TwoLeadECG', 'Coffee', 'Cricket_Y', 'SonyAIBORobotSurface',
       'NonInvasiveFatalECG_Thorax2', 'FacesUCR', 'MiddlePhalanxTW',
       'Yoga', 'Beef', 'FacesUCR', 'DistalPhalanxOutlineAgeGroup',
       'ToeSegmentation1', 'Two_Patterns', 'DistalPhalanxOutlineAgeGroup']

# Create dataframe for layer size
df = []
layer_sizes = [8,16,32,64,128,256]
for layer_size in layer_sizes:
	for dataset in univariate_random_20:
		if dataset in done_8 and layer_size ==8:
			continue
		print "Testing on: ", dataset
		print "Layer Size: ", layer_size
		acc = plain_cnn.test_model(dataset, layer_size=layer_size)
		acc_dict = {'network': 'plain_cnn', 'layer_size': layer_size, 'dataset': dataset, 'accuracy': acc}
		df.append(acc_dict)
		dataframe = pd.DataFrame(df)
		dataframe.to_csv('layer_size_comparison.csv', index=False)
		print acc_dict


pdb.set_trace()