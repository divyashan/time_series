import pandas as pd 
import numpy as np 
import pdb

import multi_triplet_siamese_cnn 
#import triplet_siamese_cnn 

import plain_cnn 

univariate_dataset_subset = ['Gun_Point', 'FaceFour', 'Car', 'Beef', 'Coffee', 'Plane', 'BeetleFly', 'BirdChicken', 'Arrowhead', 'Herring']

# Create dataframe for max-pooling
df = []
stride_vals = [ .05, .1, .2, .3, .4, .5, .7] 
for stride_val in stride_vals:
	# Only perform one of these across all max pooling values 
	for dataset in univariate_dataset_subset:
		print 'STRIDE VAL: ', stride_val
		print 'DATASET: ', dataset, '\n'
		acc = plain_cnn.test_model(dataset, stride_pct=stride_val)
		df.append({'network': 'triplet_siamese_cnn', 'stride_pct': stride_val, 'dataset': dataset, 'accuracy': acc})

		dataframe = pd.DataFrame(df)
		dataframe.to_csv('stride_size_comparison.csv', index=False)

pdb.set_trace()