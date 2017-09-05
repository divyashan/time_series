import pdb
import sys

datasets = ['Lighting7', 'Gun_Point', 'FaceFour', 'Car', 'Beef', 'Coffee', 'Plane', 'BeetleFly', 'BirdChicken', 'Arrowhead', 'Herring']

def scrape_accuracies(datasets, input_f, output_f):
    f = open(input_f, 'r').readlines()
    accuracies = [line for line in f if 'Accuracy' in line]
    tr_accs = [line for line in accuracies if 'training' in line]
    tr_accs = [line.split(' ')[-1][:-1] for line in tr_accs]
    te_accs = [line for line in accuracies if 'test' in line]
    te_accs = [line.split(' ')[-1] for line in te_accs]
    actual_te_accs = [line for line in accuracies if 'NN' in line]
    actual_te_accs = [line.split(' ')[-1] for line in actual_te_accs]
    actual_te_accs = [actual_te_accs[i*2-1] for i in range(1,12)]

    acc_output = open(output_f + "_accs.txt", 'w')
    tr_te_output = open(output_f+"_training_testing.txt", 'w')
    pdb.set_trace()
    for i, accs in enumerate(actual_te_accs):
        acc_output.write(datasets[i] + "\t" + accs) 
    """
    for i, accs in enumerate(zip(tr_accs, te_accs)):
        tr_te_output.write(datasets[i] + "\t" + accs[0] + "\t" + accs[1])
    """
def scrape_accuracies_full_test(datasets, input_f, output_f):
    n_tests = 10
    f = open(input_f, 'r').readlines()
    accuracies = [line for line in f if 'Accuracy' in line]
    tr_accs = [line for line in accuracies if 'training' in line]
    tr_accs = [line.split(' ')[-1][:-1] for line in tr_accs]
    te_accs = [line for line in accuracies if 'test' in line]
    te_accs = [line.split(' ')[-1] for line in te_accs]
    actual_te_accs = [line for line in accuracies if 'NN' in line]
    actual_te_accs = [line.split(' ')[-1][:-1] for line in actual_te_accs]
    acc_output = open(output_f + "_accs.txt", 'w')
    tr_te_output = open(output_f+"_training_testing.txt", 'w')
    for i in range(len(actual_te_accs)/10):
        dataset_accs = actual_te_accs[10*i:10*i+10]
        all_te_accs = "\t".join(dataset_accs)
        acc_output.write(datasets[i] + "\t" + all_te_accs + "\n")   
   
    for i in range(len(tr_accs)/10):
        tr_te_output.write(datasets[i] + "\t" + tr_accs[10*i] + "\t" + te_accs[10*i])    
    
scrape_accuracies(datasets, sys.argv[1], sys.argv[2])
