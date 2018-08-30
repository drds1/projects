import numpy as np
import code_pythonsubs as cpc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.feature_extraction import FeatureHasher as fhe
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from sklearn.externals import joblib

os.system('python prep_codes.py')

ytit = 'Predicted Spend ($)' 
xtit = 'Actual Spend ($)'
figtit = 'Predicted vs Actual Spend'

retrain = 0
subsamp_trial = 50000
#!!!!!!!!!!LOAD AND PREPROCESS DATA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#load the shopping data using pandas
features = pd.read_csv('BlackFriday.csv',sep=',',keep_default_na=False)
if (subsamp_trial > 0):
 features = features.iloc[:subsamp_trial] #practise on a small sample before applying to full data set to sdav etime

feature_list = list(features.columns)


#first need an array of indices of which variables to treat as categorical
row1 = features.head(1).values[0,:]
ncol = np.shape(row1)[0]
catlist = [i for i in range(ncol) if not np.str(row1[i]).isdigit()]
fconv = [feature_list[i] for i in catlist]

#one hot encoding of categorical features Note that this changes the order of the 
#attributes to put categorical values at the right end of the data matrix
fnew = pd.get_dummies(features, prefix=None, prefix_sep='_', dummy_na=False, columns=fconv, sparse=False, drop_first=False)

#remove bad data (nans, infinities etc)
fnew = fnew.convert_objects(convert_numeric=True)
fnew = fnew.replace([np.inf, -np.inf], np.nan)
fnew = fnew.dropna(how='any')



#extract the labels from the data frame 
labels = np.array(fnew[['Purchase']].values)[:,0]


#drop the sale price (label) and the order (unimportant indexing) from the data matrix
fnew = fnew.drop(['Purchase'],axis =1)



# Update the feature list for the post processed data (post one hot encoding) 
feature_list = list(fnew.columns)

#convert the data matrix to numpy array
print 'converting to array'
fnew = np.array(fnew)
print 'done converting to array'
#The data should now all be loaded into appropriate form with one hot binary encoding
#of categorical variables




 



ndat = np.shape(fnew[:,0])[0]
#exctract a sample of test data for cross validation
itest = np.random.choice(np.arange(ndat),size=1000,replace=False)
ifull = np.arange(ndat)

id = np.array(list(set(itest) & set(ifull)))
#id = np.where(ifull == itest)[0]
itrain = np.delete(ifull,id)
ftest = fnew[itest,:]
ltest = labels[itest]
ftrain = fnew[itrain,:]
ltrain = labels[itrain]










#new step for the black friday shopping data. The dataset is too large. Need to split data set
#into smaller sizes for batch learning to prevent computer from trying to load it all into
#memory in one go. As well as having new trees to fit a newmodel on the same data
#the forrest will also add trees for each sub_dataset. This is called batch learning.
#done for very large sample sizes that cannot simultaneously be loaded into memory
# split your data into an iterable of (X,y) pairs
# size each one so that it can fit into memory
maxsize = 25000
ndat = np.shape(ftrain[:,0])[0]
nsplit = np.int(np.ceil(ndat/maxsize))
data_splits = []
for i in range(nsplit):
 ilo = i*maxsize
 ihi = np.min([ndat,ilo+maxsize])
 data_splits.append([fnew[ilo:ihi,:],labels[ilo:ihi]])

#clf = RandomForestClassifier(warm_start = True, n_estimators = 1)
from sklearn.ensemble import RandomForestRegressor
if (retrain == 1):
 rf = RandomForestRegressor(warm_start=True,n_estimators= 1, random_state=42,n_jobs=-1)

 for i in range(10): # 10 passes through the data
     idsplit = 1
     for X, y in data_splits: 
         print 'batch learn sample',idsplit,'of',nsplit,' pass...',i
         rf.fit(X,y)
         rf.n_estimators += 1 # increment by one so next  will add 1 tree
         idsplit = idsplit + 1
  
 #save the trained model so we can reuse for new test data without having to retrain       
 joblib.dump(rf,'trained_forrest_blackfriday.pkl')





rf = joblib.load('trained_forrest_blackfriday.pkl') 
#now test the model on the 'test' training data for cross validation



#
#
#
#
#
##!########## Split the data into training and testing sets for cross validation #########
#
##now convert all labelled attributes to array (no need)
#from sklearn.model_selection import train_test_split
#
#
#train_features, test_features, train_labels, test_labels = \
#train_test_split(fnew, labels, test_size = 0.25,random_state = 42)
#
#
#
#print('Training Features Shape:', train_features.shape)
#print('Training Labels Shape:', train_labels.shape)
#print('Testing Features Shape:', test_features.shape)
#print('Testing Labels Shape:', test_labels.shape)
#
#
#
test_features = ftest
test_labels   = ltest
train_features = ftrain
train_labels = ltrain
#
#
i = 0
iv = 0
#use cross validation to assess the model accuracy
t0 = time.time()
predictions = rf.predict(test_features)
t1 = time.time()
# Make predictions and determine the error
errors = abs(predictions - test_labels)
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'dollars')
accuracy = 100 - np.mean(mape)
print('Accuracy random forrest:', np.round(accuracy, 2), '%.')
print ''



#make a plot of the predicted vs actual values
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(test_labels,predictions,c='r',s=2,label = 'data points')
idsort = np.argsort(test_labels)
pr = predictions[idsort]
tl = test_labels[idsort]
er = errors[idsort]
#include running rms
nres = 100
pmin, pmax = np.min(pr),np.max(pr)
pres = np.linspace(pmin,pmax,nres)
lmin, lmax = np.min(tl),np.max(tl)
lres = np.linspace(lmin,lmax,nres)
eres = np.interp(pres,tl,er)
ares = np.interp(pres,pr,tl)

try:
 fit_coef,sig_coef,xorth = cpc.polyfit(tl,pr,er,1)
 sig_coef = sig_coef
 xplot = lres
 yplot = fit_coef[1]*(xplot - xorth) + fit_coef[0]
 sigplot = np.sqrt((xplot-xorth)**2*sig_coef[1]**2 + sig_coef[0]**2)
 ax1.plot(xplot,yplot,label='least-squares fit')
 ax1.fill_between(xplot,yplot-sigplot,yplot+sigplot,alpha=0.4,label=None)
except:
 pass
#ax1.fill_between(pres,ares-eres,ares+eres,alpha=0.3)
#fit a polynomial to determine the error bars

ax1.set_xlabel(xtit)
ax1.set_ylabel(ytit)
xlim = list(ax1.get_xlim())
ylim = list(ax1.get_ylim())
ax1.set_title(figtit)
ax1.plot(xlim,ylim,ls='--',color='k',label='one-to-one line')
plt.legend()
plt.savefig('fig_actual_vs_predict.pdf')
plt.clf()

#Visualizing a Single Decision Tree
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot

# Pull out one tree from the forest
tree = rf.estimators_[5]

# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True)# precision = 1)

# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')

# Write graph to a png file
graph.write_png('tree.png')

print('The depth of this tree is:', tree.tree_.max_depth)




#Smaller tree for visualization.
# Limit depth of tree to 2 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3, random_state=42)
rf_small.fit(train_features[:1000,:], train_labels[:1000])

# Extract the small tree
tree_small = rf_small.estimators_[5]

# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True)#, precision = 1)

(graph, ) = pydot.graph_from_dot_file('small_tree.dot')

graph.write_png('small_tree_'+np.str(i)+'_'+np.str(iv)+'.png');







#!!!!!!!!!! VARIABLE IMPORTANCES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# Get numerical feature importances
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
#[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]




#!!!!!!!!!! MODEL WITH TWO MOST IMPORTANT FEATURES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)

# Extract the two most important features
important_indices = [feature_list.index(feature_importances[0][0]), feature_list.index(feature_importances[1][0])]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]

# Train the random forest
rf_most_important.fit(train_important, train_labels)

# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'dollars')
accuracy = 100 - mape
print('Accuracy random forrest 2 important:', np.round(accuracy, 2), '%.')






#!!!!!!!!!! vizualisations !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#variable importances
# list of x locations for plotting
importances = np.array(importances)
idsort = np.argsort(importances)[::-1]
nplot = 10
imp = importances[idsort][:nplot]
fl = [feature_list[ids] for ids in idsort][:nplot]

x_values = list(range(len(imp)))

# Make a bar chart
plt.bar(x_values, imp, orientation = 'vertical',color='r')

# Tick labels for x axis
plt.xticks(x_values, fl, rotation='vertical')

# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')


plt.tight_layout()
plt.savefig('importances_'+np.str(i)+'_'+np.str(iv)+'.png')

