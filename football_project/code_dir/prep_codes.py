import os
import matplotlib.pylab as plt
import sklearn
import numpy as np

psubs = ['/Users/ds207/Documents/standrews/sta/python/PCA_training/pca_custom.py',
'/Users/ds207/Documents/standrews/sta/python/k_nearest_n/k_nn.py',
'/Users/ds207/Documents/standrews/sta/python/kmeans_cluster/py_km.py',
'/Users/ds207/Documents/standrews/sta/python/neural_net_training/back_prop_eg/my_neural_net.py',
'/Users/ds207/Documents/standrews/sta/python/neural_net_training/nn_plot.py']
#'/Users/ds207/Documents/standrews/sta/python/neural_net_training/back_prop_eg/nn_eg2_new.py']



os.system('rm code_pythonsubs.py')
with open('code_pythonsubs.py', "wb") as outfile:
 for f in psubs:
  with open(f, "rb") as infile:
   a = infile.read()
   outfile.write(a)
   outfile.write('\n')
   outfile.write('\n')
   outfile.write('\n')
   outfile.write('\n')

 
