# models directory
* plain_cnn.py : vanilla CNN classifier for time series
* multi_cnn.py : multivariate form of CNN from before (only changes to the way the batches are taken in); should also take in univariate time series, but haven't tested it 

* siamese_mlp.py : siamese MLP 
* siamese_cnn.py : siamese CNN

* triplet_siamese_mlp.py : triplet form of siamese_mlp.py
* triplet_siamese_cnn.py : triplet form of siamese_cnn.py 


* cnn_helper : contains CNN architecture used in siamese CNN (main_cnn.py) and triplet CNN (triplet_siamese_cnn.py) 
* utils : contains functions for triplet creation, triplet index creation, filter plotting, and evaluating the accuracy of a given testing set embedding 
* dist_scatter_plot.py : creates a plot that visualizes the distribution of pairwise distances and and labels them based on if both points are from the same class or not
* scrape_accuracies.py : takes the output of runs of these files and creates a TSV file with datasets and their accuracies
