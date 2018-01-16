# Jiffy: A Convolutional Approach to Time Series Similarity

Jiffy is a lightweight CNN that produces a data-dependent embedding for a time series dataset. Follow the testing instructions below to reproduce the paper's results.

## Testing Instructions

We test the implementation on 6 publicly available datasets: 
* arabic_digits:  [access here](http://archive.ics.uci.edu/ml/machine-learning-databases/00195/)
* AUSLAN:         [access here](https://archive.ics.uci.edu/ml/machine-learning-databases/auslan2-mld/auslan.data.html)
* character_trajectories: [access here](https://archive.ics.uci.edu/ml/datasets/Character+Trajectories)
* libras: [access here](https://archive.ics.uci.edu/ml/datasets/Libras+Movement)
* ecg: [access here](http://www.cs.cmu.edu/~bobski/data/data.html)
* wafer: [access here](http://www.cs.cmu.edu/~bobski/data/data.html)



But there's no need to download them all separately! Create the datasets directory by running the following script in the root directory:

```
./download_datasets.sh
```

You can then run the evaluation of Jiffy on all 6 datasets with the following command:

```
python models/test_scripts/test_jiffy.py
```

