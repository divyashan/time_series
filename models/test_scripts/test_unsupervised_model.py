import pandas as pd
import sys
import pdb


sys.path.insert(0, '../')
from time_series.models import ts_autoencoder
from time_series.LDPS import clustering
from time_series.models import kshape


from time_series.models.utils import MV_DATASETS


methods = [(ts_autoencoder.test_model, 'TS_autoencoder'), (clustering.test_model, 'LDPS'), (kshape.test_model, 'K-Shape')]

results_dict = []
for test_f,test_name in methods:
	print '===================================='
	dataset = 'wafer'
	print 'Function: ', test_name
	print 'Dataset: ', dataset
	RI, train_time, inf_time = test_f(dataset)
	result = dict()
	result['method'] = test_name
	result['RI'] = RI
	result['dataset'] = dataset
	result['train_time'] = train_time
	result['inf_time'] = inf_time

	results_dict.append(result)


pdb.set_trace()
results_df = pd.DataFrame(results_dict)
results_df.to_csv('unsupervised_results.csv')