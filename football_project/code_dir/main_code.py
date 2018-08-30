import numpy as np
import code_pythonsubs as cpc
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import time
import os



os.system('rm *.pyc')
os.system('python prep_codes.py')

#load data and arrange into suitable format
#!!!!!!!!!! DATA PREPARATION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ONE HOT ENCODING

features = pd.read_csv('train.csv')


features = features.replace(['NULL'], np.nan).dropna(how="any")
#drop the date
#features.drop(['Date'],axis=1)

ndat,ncol = features.shape
#convert date into either numerical value or day of the week or both
#maybe performance is function of time of year or of day of week
date_str = features.values[:,2]
date_dt  = pd.to_datetime(date)
day_count_year = [(date_dt[i] - pd.Timestamp(year=dt[i].year, month=1, day=1)).days for i in range(ndat)]
dow = [np.int(date_dt[i].weekday()) for i in range(ndat)]

fnew = features.assign(dow=dow)
fnew = fnew.assign(day_count=day_count_year)






ndat,ncol = features.shape


#Index([u'MatchId', u'LeagueId', u'Date', u'HomeTeamId', u'AwayTeamId',
#       u'Home_Goals', u'Away_Goals', u'Home_Corners', u'Away_Corners', u'dow',
#       u'day_count'],
#      dtype='object')


#model too complex to feed in all material into giant nn. Simplifications
# add together home and away teams and study how score depends on number of corners
#then study if this changes by league
# then introduce competition and classify by individual leagues


#plots goals vs corners
fig = plt.figure()
ax1 = fig.add_subplot(111)
xh,xa = np.array(fnew.values[:,7],dtype='float'),np.array(fnew.values[:,8],dtype='float')
yh,ya = np.array(fnew.values[:,5],dtype='float'),np.array(fnew.values[:,6],dtype='float')
#x = np.concatenate([fnew.values[:,7],fnew.values[:,8]])
#y = np.concatenate([fnew.values[:,5],fnew.values[:,6]])
ax1.scatter(xh,yh,label='home')
ax1.scatter(xa,ya,label='away')
ax1.set_xlabel('Number of corners')
ax1.set_ylabel('Number of goals')
ax1.legend()
plt.savefig('fig_golvscorner.png')


fig = plt.figure()

binss = bins=[np.linspace(-1,12,14),np.linspace(-1,9,11)]

ax1 = fig.add_subplot(211)
pic = ax1.hist2d(xh,yh,label='home',bins=binss,cmap='Greens')
ax1.set_ylabel('home goals')
ax1.set_xlabel('home corners')
fig.colorbar(pic[3],label='Number of matches')
#nudge boxes by half a unit
xt = list(ax1.get_xticks())
xt = [xti+0.5 for xti in xt]
ax1.set_xticks(xt)
ax1.set_xticklabels([np.str(np.int(xti-0.5)) for xti in xt])
yt = list(ax1.get_yticks())
yt = [yti+0.5 for yti in yt]
ax1.set_yticks(yt)
ax1.set_yticklabels([np.str(np.int(yti-0.5)) for yti in yt])



ax1 = fig.add_subplot(212)
pic = ax1.hist2d(xa,ya,label='away',bins=binss,cmap='Reds')
ax1.set_ylabel('away goals')
ax1.set_xlabel('away corners')
fig.colorbar(pic[3],label='Number of matches')
#nudge boxes by half a unit
xt = list(ax1.get_xticks())
xt = [xti+0.5 for xti in xt]
ax1.set_xticks(xt)
ax1.set_xticklabels([np.str(np.int(xti-0.5)) for xti in xt])
yt = list(ax1.get_yticks())
yt = [yti+0.5 for yti in yt]
ax1.set_yticks(yt)
ax1.set_yticklabels([np.str(np.int(yti-0.5)) for yti in yt])


plt.tight_layout()
plt.savefig('fig_golvscorner_hist.png')


xtot = xa + xh
ytot = ya + yh

#xmin= np.min(xtot)
#xmax = np.max(xtot)
#ymin = np.min(ytot)
#ymax = np.max(ytot)
#xi = np.linspace(xmin, xmax, nx)
#yi = np.linspace(ymin, ymax, ny)
#zi = ml.griddata(xtot, ytot, z, xi, yi)

fig = plt.figure()
ax1 = fig.add_subplot(111)
nleague = np.unique(fnew.values[:,1])
for fn in nleague:
 idx = np.where(fnew.values[:,1] == fn)[0]
 dat = np.histogram2d(xtot,ytot,bins=binss)
 ax1.contour(dat[0],extent=[np.min(dat[1]),np.max(dat[1]),np.min(dat[2]),np.max(dat[2])])#,levels=1)
ax1.set_xlabel('combined numnber of corners')
ax1.set_ylabel('combined number of goals')
plt.savefig('fig_contour.png')


#start here tomorrow asian lines https://help.smarkets.com/hc/en-gb/articles/115001324169-What-is-Asian-Handicap-betting-

#change 2d histogram into contour plot of goals vs score
#make one set of contours per league to check if the number of corners is affect by which
#league the teams are in

#need to predict the total number of corners in the match
#training set has number of goals of home and away team, number of corners of home and away team,
#date match is played, leagued and player id

#test sheet gives a line, odds that number of corners is over line, odds that number of corners is under this value
#341 units how to place them all on one match, 1 on each match etc?
#if get exactly the line number of corners, get stake returned to you

#random forrest nn. What variables
#just league


#n_class = 2

modelcomp = 1 #if 1 then do all the model comparison stuff with knn,km,rf,nn if 0- then just skip to the PCA

ntsamp = 10 #how many different sample sizes to try
nvar = 5 #how many differernt times to try a particular sample size to build up the error bar on the performance estimates







#e.g convert days of the week to ndat X 5 matrix where each day has a 1 if the correct day,
# 0's for all other days. Helps to convert multi-categorical data into binary category
# One-hot encode categorical features
#features = pd.get_dummies(features)
features.head(5)

print('Shape of features after one-hot encoding:', features.shape)
# Labels are the values we want to predict
#labels = np.array(features[ncol-1])
#convert clss labels into integers 
#f = np.array(pd.get_dummies(features[ncol-1]))
#labels = np.array(f[:,0])
#
#
## Remove the labels from the features
## axis 1 refers to the columns
#features= features.drop(ncol-(n_class-1), axis = 1)
#
## Saving feature names for later use
#feature_list = list(features.columns)
#
## Convert to numpy array
#features = np.array(features)














#
#cpu_knn  = np.zeros((ntsamp,nvar))
#cpu_km   = np.zeros((ntsamp,nvar))
#cpu_km_pca   = np.zeros((ntsamp,nvar))
#cpu_nn   = np.zeros((ntsamp,nvar))
#cpu_nn_skl   = np.zeros((ntsamp,nvar))
#cpu_dtrf = np.zeros((ntsamp,nvar))
#
#acc_knn  = np.zeros((ntsamp,nvar))
#acc_km   = np.zeros((ntsamp,nvar))
#acc_km_pca   = np.zeros((ntsamp,nvar))
#acc_nn   = np.zeros((ntsamp,nvar))
#acc_nn_skl   = np.zeros((ntsamp,nvar))
#acc_dtrf = np.zeros((ntsamp,nvar))
#
#
#n_train = np.linspace(10,150,ntsamp,dtype='int')
#
#
## Split the data into training and testing sets, specify random_state 
## to get the same random data used in the train set each call
#
##train_features, test_features, train_labels, test_labels = \
##train_test_split(features, labels, test_size = 0.25)#random_state = 42
#
#train_f, test_features, train_l, test_labels = \
#train_test_split(features, labels, test_size = 0.25)#random_state = 42
#ntrain   = int(np.shape(n_train)[0])
#ndat_sub = np.shape(train_f[:,0])[0]
#
#
#if (modelcomp == 1):
# for i in range(ntrain):
#  
#  ntnow = n_train[i]
#  
#  for iv in range(nvar):
#   print 'training size',ntnow
#   print 'test ',iv,'of',nvar
#   
#   
#   idnow = np.random.choice(np.arange(ndat_sub), size=ntnow, replace=False, p=None)
#   train_features = train_f[idnow,:]
#   train_labels   = train_l[idnow]
#  
#   print train_features[0,:]
#   print np.shape(train_features)
#   
#   print('Training Features Shape:', train_features.shape)
#   print('Training Labels Shape:', train_labels.shape)
#   print('Testing Features Shape:', test_features.shape)
#   print('Testing Labels Shape:', test_labels.shape)
#   
#   
#   
#   
#   
#   
#   
#   
#   
#   
#   
#   
#   
#   
#   
#   #!!!!!!!!!!!!!!!!!!neural net !!!!!!!!!!!!!!!!!!!!!!
#   l_rate = 0.3
#   n_epoch = 500
#   n_hidden = 5
#   #
#   #
#   ##now to train the network on your own data
#   n_inputs = ncol-(n_class - 1)
#   n_outputs = 1#len(set([row[-1] for row in dataset]))#how many different classifications are there
#   network,idkey = cpc.train_net(train_features,train_labels,l_rate,n_epoch,n_hidden)
#   
#   t0 = time.time()
#   predictions,c_new = cpc.test_net(test_features,network)
#   t1 = time.time()
#   
#   
#   
#   #performance metric
#   ntot = np.shape(predictions)[0]
#   ncorrect = np.shape(np.where(predictions == test_labels)[0])[0]
#   accuracy = 100 * (1.*ncorrect/ntot)
#   print('Accuracy neural net:', round(accuracy, 2), '%.')
#   cpu_nn[i,iv] = t1 - t0
#   acc_nn[i,iv] = accuracy
#   
#   
#   #return the weights for network diagram plotting
#   ws = cpc.get_weights(network)
#   
#   
#   from sklearn.neural_network import MLPClassifier
#   clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5), random_state=1)
#   clf.fit(train_features, train_labels)
#   MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
#       beta_1=0.9, beta_2=0.999, early_stopping=False,
#       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
#       learning_rate_init=0.001, max_iter=200, momentum=0.9,
#       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
#       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
#       warm_start=False)
#   t0 = time.time()
#   predictions_sk = clf.predict(test_features)
#   t1 = time.time()
#      #performance metric
#      
#   ws = clf.coefs_
#   wsplot = [np.abs(ws[i2]) for i2 in range(len(ws))]
#   wsmax  = np.min(np.array([np.max(wsplot[i2]) for i2 in range(len(wsplot))]))
#   wsmin  = np.max(np.array([np.min(wsplot[i2]) for i2 in range(len(wsplot))]))
#   wsplot = [(wsplot[i2] - wsmin)/(wsmax-wsmin)/100 for i2 in range(len(wsplot))]
#   
#   
#   
#   bias = clf.intercepts_
#   bsplot = [np.abs(bias[i2]) for i2 in range(len(bias))]
#   bsmax  = np.min(np.array([np.max(bsplot[i2]) for i2 in range(len(bsplot))]))
#   bsmin  = np.max(np.array([np.min(bsplot[i2]) for i2 in range(len(bsplot))]))
#   bsplot = [(bsplot[i2] - bsmin)/(bsmax-bsmin) for i2 in range(len(bsplot))]
#   
#   nd,ndim = np.shape(test_features)
#   n_size = np.array([ndim,5,1])
#   nlayers = np.shape(n_size)[0]
#   a = cpc.makennplot(n_size,wsplot,example=0,
#   color=['k','k','k'],
#   alpha=1.0,
#   lwd=0.1,
#   horizontal_distance_between_neurons = 2,
#   vertical_distance_between_layers = 50,
#   neuron_radius = 1.0,
#   ann=['Input Layer','Neuron Output',r'$a_j^l = f \, \left( \sum_i^{N_{i}^{l-1}} W_{j}^i a_{i}^{l-1} + b_{j}^{l-1} \right)$','Hidden Layer','Output Layer'],
#   coan=[[0.0,1.0],[0.6,0.53],[0.6,0.41],[0.0,0.5],[0.0,0.0]],
#   figtit='nnplot_'+np.str(iv)+'_'+np.str(i)+'.pdf')
#   
#   #color=['r','b','k'],
#   #alpha=0.4,
#   #ann=['Input Layer','Hidden Layer','Output Layer'],
#   #coan=[[0.0,1.0],[0.0,0.5],[0.0,0.0]],
#   #vertical_distance_between_layers = 20,
#   #horizontal_distance_between_neurons = 4,
#   #neuron_radius = 1.5,figtit='nnplot_'+np.str(iv)+'_'+np.str(i)+'.pdf')
#
#   
#   #weights = [np.ones((n_size[i+1],n_size[i])).T for i in range(0,nlayers-1,1)]
#   
# 
#   
#   ntot = np.shape(predictions_sk)[0]
#   ncorrect = np.shape(np.where(predictions_sk == test_labels)[0])[0]
#   accuracy = 100 * (1.*ncorrect/ntot)
#   print('Accuracy neural net sklearn:', round(accuracy, 2), '%.')
#   cpu_nn_skl[i,iv] = t1 - t0
#   acc_nn_skl[i,iv] = accuracy
#   
#   
#   
#   
#   
#   
#   
#   
#   
#   
#   #!!!!!!!!! K_nearest neighbour #!!!!!!!!!!!!!!!
#   t0 = time.time()
#   opknn = cpc.k_test(test_features,train_features,train_labels,distance=2,k=3)
#   t1 = time.time()
#   
#   predictions = np.array(opknn)
#   ntot = np.shape(predictions)[0]
#   ncorrect = np.shape(np.where(predictions == test_labels)[0])[0]
#   accuracy = 100 * (1.*ncorrect/ntot)
#   print('Accuracy knn:', round(accuracy, 2), '%.')
#   cpu_knn[i,iv] = t1 - t0
#   acc_knn[i,iv] = accuracy
#   
#   
#   
#   
#   
#   
#   
#   
#   
#   
#   
#   
#   
#   
#   
#   #!!!!! K_means clustering #!!!!!!!
#   smean,idnow = cpc.kmeans(train_features,k=n_class,kstart=-2,nits=1000,lochange=0,diagplot=0)
#   t0 = time.time()
#   opkmeans = cpc.kmeans_predict(test_features,smean)
#   t1 = time.time()
#   
#   predictions = np.array(opkmeans)
#   ntot = np.shape(predictions)[0]
#   ncorrect = np.shape(np.where(predictions == test_labels)[0])[0]
#   accuracy = 100 * (1.*ncorrect/ntot)
#   print('Accuracy kmeans:', round(accuracy, 2), '%.')
#   cpu_km[i,iv] = t1 - t0
#   acc_km[i,iv] = accuracy
#   
#   
#
#
#   #!!!!! K_means clustering with PCA #!!!!!!!
#
#   from sklearn import decomposition
#   pca = decomposition.PCA(n_components=2)
#   train_f_new = (train_features - np.mean(train_features,axis=0))/np.std(train_features,axis=0)
#   pca.fit(train_f_new)
#   X = pca.transform(train_f_new)
#   test_f_new = (test_features - np.mean(test_features,axis=0))/np.std(test_features,axis=0)
#   X_test = pca.transform(test_f_new)
#
#
#   smean,idnow = cpc.kmeans(train_f_new,k=n_class,kstart=-2,nits=1000,lochange=0,diagplot=0)
#   t0 = time.time()
#   opkmeans = cpc.kmeans_predict(test_f_new,smean)
#   t1 = time.time()
#   
#   predictions = np.array(opkmeans)
#   ntot = np.shape(predictions)[0]
#   ncorrect = np.shape(np.where(predictions == test_labels)[0])[0]
#   accuracy = 100 * (1.*ncorrect/ntot)
#   print('Accuracy kmeans with pca:', round(accuracy, 2), '%.')
#   cpu_km_pca[i,iv] = t1 - t0
#   acc_km_pca[i,iv] = accuracy   
#   
#   
#   fig = plt.figure()
#   ax1 = fig.add_subplot(111)
#   idclass = np.unique(train_l)
#   nclass = np.shape(idclass)[0]
#   for ic in range(nclass):
#    idinc = np.where(train_labels == idclass[ic])[0]
#    ax1.scatter(X[idinc, 0], X[idinc, 1], label='class '+np.str(ic))
#   
#   plt.legend()
#   ax1.set_xlabel('EV 1')
#   ax1.set_ylabel('EV 2')
#   plt.savefig('PCA_twomostimportant_'+np.str(i)+'_'+np.str(iv)+'.pdf')
#   
#   
#   
#   
#   
#   #!!!!!!!!!! RANDOM FORREST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#   
#   #!!!!!!!!!!TRAIN MODEL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#   
#   # Import the model we are using Classifier in this case
#   from sklearn.ensemble import RandomForestClassifier
#   
#   # Instantiate model 
#   rf = RandomForestClassifier(n_estimators= 1000, random_state=42)
#   
#   # Train the model on training data
#   rf.fit(train_features, train_labels);
#   
#   
#   
#   #!!!!!!!!!! MAKE PREDICTIONS ON TEST DATA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#   #!!!!!!!!!! DETERMINE PERFORMANCE METRICS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#   t0 = time.time()
#   predictions = rf.predict(test_features)
#   t1 = time.time()
#   
#   ntot = np.shape(predictions)[0]
#   ncorrect = np.shape(np.where(predictions == test_labels)[0])[0]
#   accuracy = 100 * (1.*ncorrect/ntot)
#   print('Accuracy Random Forrest:', round(accuracy, 2), '%.')
#   cpu_dtrf[i,iv] = t1 - t0
#   acc_dtrf[i,iv] = accuracy
#   
#   #!!!!!!!!!! INTERPRET MODEL RESULTS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#   
#   
#   #Visualizing a Single Decision Tree
#   # Import tools needed for visualization
#   from sklearn.tree import export_graphviz
#   import pydot
#   
#   # Pull out one tree from the forest
#   tree = rf.estimators_[5]
#   
#   # Export the image to a dot file
#   export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
#   
#   # Use dot file to create a graph
#   (graph, ) = pydot.graph_from_dot_file('tree.dot')
#   
#   # Write graph to a png file
#   graph.write_png('tree.png')
#   
#   print('The depth of this tree is:', tree.tree_.max_depth)
#   
#   
#   
#   
#   
#   #Smaller tree for visualization.
#   # Limit depth of tree to 2 levels
#   rf_small = RandomForestClassifier(n_estimators=10, max_depth = 3, random_state=42)
#   rf_small.fit(train_features, train_labels)
#   
#   # Extract the small tree
#   tree_small = rf_small.estimators_[5]
#   
#   # Save the tree as a png image
#   export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
#   
#   (graph, ) = pydot.graph_from_dot_file('small_tree.dot')
#   
#   graph.write_png('small_tree_'+np.str(i)+'_'+np.str(iv)+'.png');
#   
#   
#   
#   
#   
#   
#   
#   #!!!!!!!!!! VARIABLE IMPORTANCES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#   
#   
#   # Get numerical feature importances
#   importances = list(rf.feature_importances_)
#   
#   # List of tuples with variable and importance
#   feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
#   
#   # Sort the feature importances by most important first
#   feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
#   
#   # Print out the feature and importances 
#   #[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
#   
#   
#   
#   
#   #!!!!!!!!!! MODEL WITH TWO MOST IMPORTANT FEATURES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#   # New random forest with only the two most important variables
#   rf_most_important = RandomForestClassifier(n_estimators= 1000, random_state=42)
#   
#   # Extract the two most important features
#   important_indices = [feature_list.index(feature_importances[0][0]), feature_importances[1][0]]
#   train_important = train_features[:, important_indices]
#   test_important = test_features[:, important_indices]
#   
#   # Train the random forest
#   rf_most_important.fit(train_important, train_labels)
#   
#   # Make predictions and determine the error
#   predictions = rf_most_important.predict(test_important)
#   
#   #errors = abs(predictions - test_labels)
#   ntot = np.shape(predictions)[0]
#   ncorrect = np.shape(np.where(predictions == test_labels)[0])[0]
#   # Display the performance metrics
#   #print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
#   
#   accuracy = 100 * (1.*ncorrect/ntot)
#   #accuracy = 100 - mape
#   
#   print('Accuracy random forrest 2 important:', round(accuracy, 2), '%.')
#   
#   
#   
#   #!!!!!!!!!! vizualisations !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#   
#   #variable importances
#   # list of x locations for plotting
#   x_values = list(range(len(importances)))
#   
#   # Make a bar chart
#   plt.bar(x_values, importances, orientation = 'vertical')
#   
#   # Tick labels for x axis
#   plt.xticks(x_values, feature_list, rotation='vertical')
#   
#   # Axis labels and title
#   plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')
#   
#   
#   
#   plt.savefig('importances_'+np.str(i)+'_'+np.str(iv)+'.png')
#
#
#
##use PCA to extract the two most important features from the data set and use these as the classifiers
##kfeat = 2
##
####test my PCA code by comparing to skleanr PCA
###ndim = 2
###cov = np.diag(np.ones(ndim))
###mean = np.zeros(ndim)
###cov[0,1] = 1.0
####cov[2,0] = 1.0
###cov[0,0] = 4.0
####random data multivariate gaussian
###train_f = np.random.multivariate_normal(mean,cov,50000)
##
##
##
###evec,eval,fraceval,idsort,newdat,newdatsort,mean,std = cpc.mypca(train_f,kfeat,diagfolder='',label=[],meannorm = 1,stdnorm=1)
###features_new = newdatsort[:,:kfeat]
###compare this PCA with sklearn PCA to check mine works!
##from sklearn import decomposition
##pca = decomposition.PCA(n_components=2)
##
##train_f_new = (train_f - np.mean(train_f,axis=0))/np.std(train_f,axis=0)
##
##pca.fit(train_f_new)
##X = pca.transform(train_f_new)
##
##fig = plt.figure()
##ax1 = fig.add_subplot(211)
##
##
##ax2 = fig.add_subplot(212)
##
##idclass = np.unique(train_l)
##nclass = np.shape(idclass)[0]
##for i in range(nclass):
## idinc = np.where(train_l == idclass[i])[0]
## 
## ax1.scatter(features_new[idinc,0],features_new[idinc,1],label='class '+np.str(i))
## ax2.scatter(X[idinc, 0], X[idinc, 1], label='class '+np.str(i))
##
##plt.legend()
##ax1.set_xlabel('EV 1 (mine)')
##ax1.set_ylabel('EV 2 (mine)')
##ax2.set_xlabel('EV 1 (sklearn)')
##ax2.set_ylabel('EV 2 (sklearn)')
##plt.savefig('PCA_twomostimportant.pdf')
##
###start fromhere tomorrow
##
##print 'WHY IS THE PSKLEARN PCA giving different results to mine (is it because I just changed my PCA code to divide by the standard deviation as well as subgtract the mean)'
##raw_input()
#
#
#
#
#
#
#
#
#
##compute the comparison performance metrics for the four algorithms
#cpu_ave_dtrf = np.mean(cpu_dtrf,axis=1)
#cpu_ave_nn = np.mean(cpu_nn,axis=1)
#cpu_ave_nn_skl = np.mean(cpu_nn_skl,axis=1)
#cpu_ave_km = np.mean(cpu_km,axis=1)
#cpu_ave_km_pca = np.mean(cpu_km_pca,axis=1)
#cpu_ave_knn = np.mean(cpu_knn,axis=1)
#acc_ave_dtrf = np.mean(acc_dtrf,axis=1)
#acc_ave_nn = np.mean(acc_nn,axis=1)
#acc_ave_nn_skl = np.mean(acc_nn_skl,axis=1)
#acc_ave_km = np.mean(acc_km,axis=1)
#acc_ave_km_pca = np.mean(acc_km_pca,axis=1)
#acc_ave_knn = np.mean(acc_knn,axis=1)
#
#cpu_sig_dtrf = np.std(cpu_dtrf,axis=1)
#cpu_sig_nn = np.std(cpu_nn,axis=1)
#cpu_sig_nn_skl = np.std(cpu_nn_skl,axis=1)
#cpu_sig_km = np.std(cpu_km,axis=1)
#cpu_sig_km_pca = np.std(cpu_km_pca,axis=1)
#cpu_sig_knn = np.std(cpu_knn,axis=1)
#acc_sig_dtrf = np.std(acc_dtrf,axis=1)
#acc_sig_nn = np.std(acc_nn,axis=1)
#acc_sig_nn_skl = np.std(acc_nn_skl,axis=1)
#acc_sig_km = np.std(acc_km,axis=1)
#acc_sig_km_pca = np.std(acc_km_pca,axis=1)
#acc_sig_knn = np.std(acc_knn,axis=1)
#
#
##make plots for the performance comparisons
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.errorbar(n_train,cpu_ave_dtrf,cpu_sig_dtrf,marker='o',label='Random Forrest')
##ax1.errorbar(n_train,cpu_ave_nn,cpu_sig_nn,marker='o',label='Neural Network')
#ax1.errorbar(n_train,cpu_ave_nn_skl,cpu_sig_nn_skl,marker='o',label='Neural Network')
#ax1.errorbar(n_train,cpu_ave_km,cpu_sig_km,marker='o',label='K-means Cluster')
#ax1.errorbar(n_train,cpu_ave_km_pca,cpu_sig_km_pca,marker='o',label='K-means Cluster (PCA)')
#ax1.errorbar(n_train,cpu_ave_knn,cpu_sig_knn,marker='o',label='K_nearest-neighbour')
#ax1.set_xlabel('training sample size')
#ax1.set_ylabel('computation time (seconds)')
#ax1.set_yscale('log')
#plt.legend()
#plt.savefig('comparison_time.pdf')
#
##plot for accuracy comparison
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.errorbar(n_train,acc_ave_dtrf,acc_sig_dtrf,marker='o',label='Random Forrest')
##ax1.errorbar(n_train,acc_ave_nn,acc_sig_nn,marker='o',label='Neural Network')
#ax1.errorbar(n_train,acc_ave_nn_skl,acc_sig_nn_skl,marker='o',label='Neural Network')
#ax1.errorbar(n_train,acc_ave_km,acc_sig_km,marker='o',label='K-means Cluster')
#ax1.errorbar(n_train,acc_ave_km_pca,acc_sig_km_pca,marker='o',label='K-means Cluster (PCA)')
#ax1.errorbar(n_train,acc_ave_knn,acc_sig_knn,marker='o',label='K_nearest-neighbour')
#ax1.set_xlabel('training sample size')
#ax1.set_ylabel('accuracy (%)')
#plt.legend()
#plt.savefig('comparison_accuracy.pdf')
#
#
#
#
#
#
#