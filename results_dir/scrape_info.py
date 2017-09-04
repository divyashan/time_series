import sys
import pdb

def write_to_standard_format(dataset_file, fout):
	results = open(dataset_file, 'r').readlines()
	results = [dataline.split('\t') for dataline in results]
	dataset_names = [[dataline[0], 1-float(dataline[-1][:-1])] for dataline in results]
	dataset_names = sorted(dataset_names)
	output = open(fout, 'w')
	for result in dataset_names:
		output.write(result[0] + "\t" + str(result[1]) + "\n")

write_to_standard_format(sys.argv[1], sys.argv[2])
