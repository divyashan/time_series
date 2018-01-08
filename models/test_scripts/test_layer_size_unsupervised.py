import pandas as pd 
import numpy as np 
import pdb

import ts_autoencoder 
from time_series.models.utils import MV_DATASETS
#import triplet_siamese_cnn 


done_8 = ['Lighting2', 'uWaveGestureLibrary_X', 'MoteStrain','ECG200', 'ProximalPhalanxOutlineAgeGroup',
       'TwoLeadECG', 'Coffee', 'Cricket_Y', 'SonyAIBORobotSurface', 'uWaveGestureLibrary_Z',
       'NonInvasiveFatalECG_Thorax2', 'FacesUCR', 'MiddlePhalanxTW',
       'Yoga', 'Beef', 'FacesUCR', 'DistalPhalanxOutlineAgeGroup',
       'ToeSegmentation1']

univariate_random_20 = ['Lighting2', 'MoteStrain', 'ECG200',
       'TwoLeadECG', 'Coffee', 'Cricket_Y',
       'NonInvasiveFatalECG_Thorax2', 'FacesUCR', 'MiddlePhalanxTW',
       'Yoga', 'Beef', 'FacesUCR', 'DistalPhalanxOutlineAgeGroup',
       'ToeSegmentation1', 'Two_Patterns', 'DistalPhalanxOutlineAgeGroup']



univariate_random = ['Beef', 'Yoga', 'Coffee', 'ECG200', 'FacesUCR', 'Cricket_Y', 'Lighting2', 'TwoLeadECG', 'Two_Patterns', 'MoteStrain']

# Create dataframe for layer size
df = []
layer_pctgs = [i for i in np.arange(0, 50, 5)]
layer_pctgs_other = [j for j in np.arange(50, 250, 20)]
layer_pctgs.extend(layer_pctgs_other)
out = open('trash', 'w')
for layer_size_pctg in layer_pctgs:
	for dataset in MV_DATASETS:
		print
		print "Testing on: ", dataset
		print "Layer Size: ", layer_size_pctg
		ri, _, _ = ts_autoencoder.test_model(dataset, embedding_size_pctg=layer_size_pctg)
		acc_dict = dict()
		acc_dict = {'network': 'ts_autoencoder', 'size': layer_size_pctg, 'dataset': dataset, 'accuracy': ri}
		df.append(acc_dict)
		dataframe = pd.DataFrame(df)
		print 'finished: ', acc_dict
		out.write(str(layer_size_pctg) + "\t" + dataset + "\t" + str(ri))
		dataframe.to_csv('layer_size_unsupervised_comparison_size.csv', index=False)


