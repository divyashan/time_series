import pandas as pd 
import numpy as np 
import pdb
import sys

sys.path.insert(0, '../')


import multi_cnn

from time_series.models.utils import MV_DATASETS


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
	for dataset in MV_DATASETS:
		print "Testing on: ", dataset
		print "Layer Size: ", layer_size
		acc = multi_cnn.test_model(dataset, .1, layer_size)
		acc_dict = {'network': 'multi_cnn', 'size': layer_size, 'dataset': dataset, 'accuracy': acc}
		df.append(acc_dict)
		dataframe = pd.DataFrame(df)
		dataframe.to_csv('layer_size_comparison.csv', index=False)
		print acc_dict


pdb.set_trace()