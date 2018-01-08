import pandas as pd 
import numpy as np 
import pdb

import multi_triplet_siamese_cnn 
#import triplet_siamese_cnn 

import plain_cnn 
#import multi_cnn 

#import triplet_siamese_mlp 
#import siamese_mlp 
#import siamese_cnn




#networks = {'siamese_MLP': siamese_mlp.test_model, 'triplet_siamese_MLP': triplet_siamese_mlp.test_model, 
#			'plain_CNN': plain_cnn.test_model, 'siamese_CNN':siamese_cnn.test_model, 'triplet_siamese_CNN': triplet_siamese_CNN.test_model}
networks = {'plain_CNN': plain_cnn.test_model}
univariate_dataset_subset = ['Gun_Point', 'FaceFour', 'Car', 'Beef', 'Coffee', 'Plane', 'BeetleFly', 'BirdChicken', 'Arrowhead', 'Herring']
univariate_dataset_all = ['50Words', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'Car', 'ChlorineConcentration', 
						  'CinC_ECG_torso', 'Coffee', 'Computers', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 'DiatomSizeReduction', 
						  'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'ECG200', 'ECG5000', 'ECGFiveDays', 
						  'Earthquakes', 'ElectricDevices', 'FaceAll', 'FaceFour', 'FacesUCR', 'Fish', 'FordA', 'FordB', 'Gun_Point', 'Ham', 
						  'HandOutlines', 'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 
						  'Lighting2', 'Lighting7', 'MALLAT', 'Meat', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 
						  'MiddlePhalanxTW', 'MoteStrain', 'Non-Invasive Fetal ECG Thorax1', 'Non-Invasive Fetal ECG Thorax2', 'OSU Leaf', 
						  'OliveOil Tony Bagnall', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup', 
						  'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType', 'ShapeletSim', 'ShapesAll', 
						  'SmallKitchenAppliances', 'SonyAIBORobot Surface', 'SonyAIBORobot SurfaceII', 'StarLightCurves', 'Strawberry', 
						  'Swedish Leaf', 'Symbols', 'Synthetic Control', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'Two Patterns', 
						  'TwoLeadECG', 'UWaveGestureLibraryAll', 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga', 
						  'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z']

univariate_dataset_all = ['FaceAll', 'FaceFour', 'FacesUCR', 'Fish', 'FordA', 'FordB', 'Gun_Point', 'Ham', 
						  'HandOutlines', 'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 
						  'Lighting2', 'Lighting7', 'MALLAT', 'Meat', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 
						  'MiddlePhalanxTW', 'MoteStrain', 'Non-Invasive Fetal ECG Thorax1', 'Non-Invasive Fetal ECG Thorax2', 'OSU Leaf', 
						  'OliveOil Tony Bagnall', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup', 
						  'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType', 'ShapeletSim', 'ShapesAll', 
						  'SmallKitchenAppliances', 'SonyAIBORobot Surface', 'SonyAIBORobot SurfaceII', 'StarLightCurves', 'Strawberry', 
						  'Swedish Leaf', 'Symbols', 'Synthetic Control', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'Two Patterns', 
						  'TwoLeadECG', 'UWaveGestureLibraryAll', 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga', 
						  'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z']

univariate_random_20 = ['Lighting2', 'uWaveGestureLibrary_Z', 'MoteStrain',
       'uWaveGestureLibrary_Z', 'ECG200', 'ProximalPhalanxOutlineAgeGroup',
       'TwoLeadECG', 'Coffee', 'Cricket_Y', 'SonyAIBORobot Surface',
       'Non-Invasive Fetal ECG Thorax2', 'FacesUCR', 'MiddlePhalanxTW',
       'Yoga', 'Beef', 'FacesUCR', 'DistalPhalanxOutlineAgeGroup',
       'ToeSegmentation1', 'Two Patterns', 'DistalPhalanxOutlineAgeGroup']

multivariate_datasets = ['EEG', 'ICU']

# Create dataframe for network comparison

df = pd.DataFrame.from_csv('network_comparison.csv')
N_TEST = 5
for dataset in univariate_dataset_all:
	print 'Starting new dataset: ', dataset, '\n\n'
	for network, method in networks.iteritems():
		accs = []
		for i in range(N_TEST):
			acc = method(dataset)
			accs.append(acc)
		df = df.append(pd.DataFrame([{'network': network, 'dataset': dataset, 'accuracy': np.max(accs)}]))
		df.to_csv('network_comparison.csv', index=False)

pdb.set_trace()	

df = []
for dataset in multivariate_datasets:
	acc = multi_triplet_siamese_cnn.test_model(dataset)
	df.append({'network': 'triplet_siamese_CNN', 'dataset': dataset, 'accuracy': acc})

# Create dataframe for max-pooling
df = []
max_pool_vals = [.05, .1, .25, .5, .75, 1.0]
for pool_val in max_pool_vals:
	# Only perform one of these across all max pooling values 
	for dataset in univariate_dataset_subset:
		acc = triplet_siamese_cnn(dataset, pool_size=pool_val)
		df.append({'network': 'triplet_siamese_cnn', 'pool_size': pool_val, 'dataset': dataset, 'accuracy': acc})



# Create dataframe for layer size
df = []
layer_sizes = [8,16,32,64,128,256,512]
for layer_size in layer_sizes:
	for dataset in univariate_dataset_subset:
		acc = plain_cnn(dataset, layer_size=layer_size)
		df.append({'network': 'triplet_siamese_cnn', 'layer_size': layer_size, 'dataset': dataset, 'accuracy': acc})

