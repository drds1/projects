from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor



class rfr:

    def __init__(self):
        '''
        specify the main timeseries (ymain)
        covariate timeseries (covariates)
        and names of each covariate(feature_list)
        '''
        self.feature_list = None
        self.covariates = None
        self.ymain = None


    def split_train_test(self):
        fnew, labels = self.covariates, self.ymain
        #!########## Split the data into training and testing sets for cross validation #########
        train_features, test_features, train_labels, test_labels = \
        train_test_split(fnew, labels, test_size = 0.25,random_state = 42)
        print('Training Features Shape:', train_features.shape)
        print('Training Labels Shape:', train_labels.shape)
        print('Testing Features Shape:', test_features.shape)
        print('Testing Labels Shape:', test_labels.shape)
        self.train_features = train_features
        self.test_features = test_features
        self.train_labels = train_labels
        self.test_labels = test_labels


    def initialize(self):
        # Instantiate model
        self.rf = RandomForestRegressor(n_estimators= 1000, random_state=42)


    def cross_validation_check(self):
        #use cross validation to assess the model accuracy
        t0 = time.time()
        predictions = self.rf.predict(test_features)
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
        self.predictions = predictions
        self.errors = errors
        self.mape = mape
        self.accuracy = accuracy


    def plot_CV_results(self):
        #make a plot of the predicted vs actual values
        test_labels = self.test_labels
        predictions = self.predictions
        errors = self.errors
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
        ax1.set_xlabel('Actual Sale Price ($)')
        ax1.set_ylabel('Predicted Sale Price ($)')
        xlim = list(ax1.get_xlim())
        ylim = list(ax1.get_ylim())
        ax1.set_title('Predicted vs Actual Sale Price')
        ax1.plot(xlim,ylim,ls='--',color='k',label='one-to-one line')
        plt.legend()
        plt.savefig('fig_actual_vs_predict.pdf')
        plt.clf()

    def plot_single_tree(self):
        #Visualizing a Single Decision Tree
        # Import tools needed for visualization
        from sklearn.tree import export_graphviz
        import pydot

        # Pull out one tree from the forest
        tree = self.rf.estimators_[5]

        # Export the image to a dot file
        export_graphviz(tree, out_file = 'tree.dot', feature_names = self.feature_list, rounded = True, precision = 1)

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






    def get_importances(self):
        #!!!!!!!!!! VARIABLE IMPORTANCES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Get numerical feature importances
        self.importances = list(self.rf.feature_importances_)

        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(self.feature_list, self.importances)]

        # Sort the feature importances by most important first
        self.feature_importances = sorted(self.feature_importances, key = lambda x: x[1], reverse = True)

        # Print out the feature and importances
        #[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]



    def two_features_only(self):
        #!!!!!!!!!! MODEL WITH TWO MOST IMPORTANT FEATURES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # New random forest with only the two most important variables
        rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)

        # Extract the two most important features
        important_indices = [self.feature_list.index(self.feature_importances[0][0]),
                             self.feature_list.index(self.feature_importances[1][0])]
        train_important = self.train_features[:, important_indices]
        test_important = self.test_features[:, important_indices]

        # Train the random forest
        rf_most_important.fit(train_important, self.train_labels)

        # Make predictions and determine the error
        predictions = rf_most_important.predict(test_important)
        errors = abs(predictions - self.test_labels)
        # Calculate mean absolute percentage error (MAPE)
        mape = 100 * (errors / self.test_labels)
        # Display the performance metrics
        print('Mean Absolute Error:', round(np.mean(errors), 2), 'dollars')
        accuracy = 100 - mape
        print('Accuracy random forrest 2 important:', np.round(accuracy, 2), '%.')





    def visualisatrions(self):
    #!!!!!!!!!! vizualisations of 2-component fit!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        #variable importances
        # list of x locations for plotting
        importances = np.array(self.importances)
        idsort = np.argsort(importances)[::-1]
        nplot = 10
        imp = importances[idsort][:nplot]
        fl = [self.feature_list[ids] for ids in idsort][:nplot]

        x_values = list(range(len(imp)))

        # Make a bar chart
        plt.bar(x_values, imp, orientation = 'vertical',color='r')

        # Tick labels for x axis
        plt.xticks(x_values, fl, rotation='vertical')

        # Axis labels and title
        plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')


        plt.tight_layout()
        plt.savefig('importances_'+np.str(i)+'_'+np.str(iv)+'.png')




if __name__ == '__main__':
    '''
    test rfr on some data
    '''
    x = rfr()
    x.split_train_test()
    x.split_train_test()
    x.initialize()
    x.cross_validation_check()
    x.plot_CV_results()
    x.plot_single_tree()
    x.get_importances()
    x.two_features_only()
    x.visualisatrions()