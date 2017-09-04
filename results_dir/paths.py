

import os

currentDir = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.dirname(currentDir)

j = os.path.join 	# abbreviation

def pathTo(subdir):
	return j(DATASETS_DIR, subdir)

MSRC_12 = pathTo(j('MSRC-12', 'origData'))
UCR = pathTo('ucr_data')
UWAVE = pathTo(j('uWave', 'extracted'))
PAMAP = pathTo('PAMAP_Dataset')
PAMAP2 = pathTo('PAMAP2_Dataset')
WARD = pathTo('WARD1.0')
