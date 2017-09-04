import subprocess
import sys

fout = open('output.txt', 'w')

datasets = ['Gun_Point', 'FaceFour', 'Car', 'Beef', 'Coffee', 'Plane', 'BeetleFly', 'BirdChicken', 'Arrowhead', 'Herring']
for dataset in datasets:
	subprocess.call([sys.executable, 'main_cnn.py', dataset],stdout=fout)
	fout.write(dataset + '\t' + dataset + '\n')


