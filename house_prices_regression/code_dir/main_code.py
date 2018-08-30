import numpy as np
import code_pythonsubs as cpc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.feature_extraction import FeatureHasher as fhe
import pandas as pd
import matplotlib.pyplot as plt
import time
import os


os.system('python prep_codes.py')



#!!!!!!!!!!LOAD AND PREPROCESS DATA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#load the house price data using pandas
features = pd.read_csv('house.dat',sep='\t',keep_default_na=False)
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
labels = np.array(fnew[['SalePrice']].values)[:,0]


#drop the sale price (label) and the order (unimportant indexing) from the data matrix
fnew = fnew.drop(['Order', 'SalePrice'],axis =1)



# Update the feature list for the post processed data (post one hot encoding) 
feature_list = list(fnew.columns)

#convert the data matrix to numpy array

fnew = np.array(fnew)

#The data should now all be loaded into appropriate form with one hot binary encoding
#of categorical variables









#!########## Split the data into training and testing sets for cross validation #########

#now convert all labelled attributes to array (no need)
from sklearn.model_selection import train_test_split


train_features, test_features, train_labels, test_labels = \
train_test_split(fnew, labels, test_size = 0.25,random_state = 42)



print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)











i=0
iv=0

#!!!!!!!!!!TRAIN MODEL USING RANDOM FORREST REGRESSOR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Import the model we are using Regressor in this case
from sklearn.ensemble import RandomForestRegressor

# Instantiate model 
rf = RandomForestRegressor(n_estimators= 1000, random_state=42)

# Train the model on training data
rf.fit(train_features, train_labels)



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
fit_coef,sig_coef,xorth = cpc.polyfit(tl,pr,er,1)
sig_coef = sig_coef
xplot = lres
yplot = fit_coef[1]*(xplot - xorth) + fit_coef[0]
sigplot = np.sqrt((xplot-xorth)**2*sig_coef[1]**2 + sig_coef[0]**2)
ax1.plot(xplot,yplot,label='least-squares fit')
ax1.fill_between(xplot,yplot-sigplot,yplot+sigplot,alpha=0.4,label=None)
#ax1.fill_between(pres,ares-eres,ares+eres,alpha=0.3)
#fit a polynomial to determine the error bars

ax1.set_xlabel('Actual Sale Price ($)')
ax1.set_ylabel('Predicted Sale Price ($)')
xlim = list(ax1.get_xlim())
ylim = list(ax1.get_ylim())
ax1.set_title('Predicted vs Actual Sale Price')
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
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)

# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')

# Write graph to a png file
graph.write_png('tree.png')

print('The depth of this tree is:', tree.tree_.max_depth)




#Smaller tree for visualization.
# Limit depth of tree to 2 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3, random_state=42)
rf_small.fit(train_features, train_labels)

# Extract the small tree
tree_small = rf_small.estimators_[5]

# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)

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

