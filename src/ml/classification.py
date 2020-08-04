import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score


def grid_search( estimator, params ):
	refit = 'accuracy'
	scoring = refit

	cv_method = RepeatedStratifiedKFold( n_splits = 10, n_repeats = 2, random_state = 0 )

	return GridSearchCV(
		estimator = estimator,
		param_grid = params,
		cv = cv_method,
		refit = refit,
		verbose = False,
		scoring = scoring,
		return_train_score = True )


def train_and_test( estimator, params, x_train, y_train,
				   x_test, y_test, plot_cmatrix = False ):

	grid = grid_search( estimator, params )

	# Training and evaluation

	grid.fit( x_train, y_train );
	print( 'Best params: ', grid.best_params_ )
	print( 'Training Accuracy', grid.best_score_ )

	y_pred = grid.best_estimator_.predict( x_test )
	print( 'Test Accuracy: ', accuracy_score( y_test, y_pred ) )

	try:
		y_pro = grid.best_estimator_.decision_function( x_test )
	except Exception as e:
		print( 'Exception arised while trying to use decision_function, predict_proba will be used instead.' )
		y_pro = grid.best_estimator_.predict_proba( x_test )
	finally:
		print( 'Test ROCauc (OvR):', roc_auc_score( y_test, y_pro, multi_class = 'ovr' ) )

	print()
	print( 'Detailed Classification Report' )
	print( classification_report( y_test, y_pred ) )
	print()

	if( plot_cmatrix ) :
		fig, ax = plt.subplots( 1, 1 )
		g = sns.heatmap( confusion_matrix( y_test, y_pred ), annot = True, cmap = "YlGnBu" )
		g.set_title( 'Test Confussion Matrix' )

	return grid


def prefix_params( d_list, prefix ):
	result = []
	for d in d_list:
		temp = {}
		for key in d.keys():
			temp[ prefix + '__' + key ] = d[ key ]
		result.append( temp )
	return result