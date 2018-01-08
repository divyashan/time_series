#!/bin/bash 

mkdir datasets

mkdir datasets/arabic-digits
wget -nd -r -np -R "index.html*" -P datasets/arabic-digits http://archive.ics.uci.edu/ml/machine-learning-databases/00195/

mkdir datasets/auslan-high-quality/tctodd
wget -nd -r -np -R "index.html*" -P datasets/auslan-high-quality https://archive.ics.uci.edu/ml/machine-learning-databases/auslan2-mld/
tar xvf datasets/auslan-high-quality/tctodd.tar.gz -C datasets/auslan-high-quality/tctodd 

mkdir datasets/char-trajectories
wget -nd -r -np -R "index.html*" -P datasets/char-trajectories https://archive.ics.uci.edu/ml/machine-learning-databases/character-trajectories/

wget -nd -r -np -R "index.html*" -P datasets/ http://www.cs.cmu.edu/~bobski/data/ecg.tar.gz
mv datasets/ecg.tar.gz datasets/ecg.tar
tar xvf datasets/ecg.tar -C ./datasets 

mkdir datasets/libras
wget -nd -r -np -R "index.html*" -P datasets/libras https://archive.ics.uci.edu/ml/machine-learning-databases/libras/

wget -nd -r -np -R "index.html*" -P datasets/ http://www.cs.cmu.edu/~bobski/data/wafer.tar.gz
mv datasets/wafer.tar.gz datasets/wafer.tar
tar xvf datasets/wafer.tar -C ./datasets

# Cleanup
rm datasets/ecg.tar
rm datasets/wafer.tar

