import os

pdir = '/microlens/ds207/python/'

psubs = ['PCA_training/pca_custom.py',
'k_nearest_n/k_nn.py',
'kmeans_cluster/py_km.py',
'neural_net_training/back_prop_eg/my_neural_net.py',
'neural_net_training/nn_plot.py',
'mystats.py']
#'/Users/ds207/Documents/standrews/sta/python/neural_net_training/back_prop_eg/nn_eg2_new.py']



os.system('rm code_pythonsubs.py')
with open('code_pythonsubs.py', "wb") as outfile:
 for f in psubs:
  with open(pdir+f, "rb") as infile:
   a = infile.read()
   outfile.write(a)
   outfile.write('\n')
   outfile.write('\n')
   outfile.write('\n')
   outfile.write('\n')

 
