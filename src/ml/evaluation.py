from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from ml.constants import *

def plot_learning_curve( estimator, title, X, y, axes = None,  ylim = None, cv = None,
						n_jobs = None, train_sizes = np.linspace( .1, 1.0, 5 ) ):

	if axes is None:
		_, axes = plt.subplots(1, 3, figsize=(20, 5))

	axes[0].set_title(title)
	if ylim is not None:
		axes[0].set_ylim( *ylim )
	axes[0].set_xlabel( 'Training examples' )
	axes[0].set_ylabel( 'Score' )

	train_sizes, train_scores, test_scores, fit_times, _ = \
		learning_curve( estimator, X, y, cv = cv, n_jobs = n_jobs,
					   train_sizes = train_sizes, return_times = True )

	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)

	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	fit_times_mean = np.mean(fit_times, axis=1)
	fit_times_std = np.std(fit_times, axis=1)

	# Plot learning curve

	axes[0].grid()
	axes[0].fill_between( train_sizes, train_scores_mean - train_scores_std,
						 train_scores_mean + train_scores_std, alpha = 0.1,
						 color = 'r')
	axes[0].fill_between( train_sizes, test_scores_mean - test_scores_std,
						 test_scores_mean + test_scores_std, alpha = 0.1,
						 color = 'g' )

	axes[0].plot( train_sizes, train_scores_mean, 'o-', color = 'r',
				 label = 'Training score' )
	axes[0].plot( train_sizes, test_scores_mean, 'o-', color = 'g',
				 label = 'Cross-validation score' )
	axes[0].legend( loc = 'best' )


	# Plot n_samples vs fit_times

	axes[1].grid()
	axes[1].plot( train_sizes, fit_times_mean, 'o-' )
	axes[1].fill_between( train_sizes, fit_times_mean - fit_times_std,
						 fit_times_mean + fit_times_std, alpha = 0.1 )
	axes[1].set_xlabel( 'Training examples' )
	axes[1].set_ylabel( 'fit_times' )
	axes[1].set_title( 'Scalability of the model' )


	# Plot fit_time vs score

	axes[2].grid()
	axes[2].plot( fit_times_mean, test_scores_mean, 'o-' )
	axes[2].fill_between( fit_times_mean, test_scores_mean - test_scores_std,
						 test_scores_mean + test_scores_std, alpha = 0.1 )
	axes[2].set_xlabel( 'fit_times' )
	axes[2].set_ylabel( 'Performance of the model' )
	axes[2].set_title( 'Performance of the model' )

	return plt


def plot_learning_curves( dfs, grids, alg_name ):
	fig, axes = plt.subplots( 3, len( dfs ), figsize=( 7 * len( dfs ), 15 ) )
	if len( axes.shape ) == 1 :
		axes = axes.reshape( 3, 1 )

	ax = 0
	for k, d in dfs:
		cv = StratifiedShuffleSplit( n_splits = 100, test_size = 0.2, random_state = 0 )
		title = 'Dataset ' + str( k )
		plot_learning_curve( grids[ k ].best_estimator_,
							title,
							d[ X_TRAIN ], d[ Y_TRAIN ],
							axes = axes[ :, ax ],
							ylim = ( 0.7, 1.01 ),
							cv = cv, n_jobs = 4 )
		ax += 1

	fig.suptitle( 'Learning Curves ' + alg_name + ' with PCA', fontsize= 20 )
	plt.show()