#import modules
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# load data
dataset = loadtxt('diabetes_sample.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data (compare logistic regression and XGboost)
model = {'XGB':XGBClassifier(),
		'Logreg':LogisticRegression(solver='lbfgs',max_iter=1000),
		'RFC':RandomForestClassifier(n_estimators=1000)}
for m in model.keys():
	model[m].fit(X_train, y_train)
	# make predictions for test data
	y_pred = model[m].predict(X_test)
	predictions = [round(value) for value in y_pred]
	# evaluate predictions
	accuracy = accuracy_score(y_test, predictions)
	print(m+": Accuracy: %.2f%%" % (accuracy * 100.0))
