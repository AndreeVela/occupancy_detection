import pandas as pd
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif



def select_k_rfe( x, y, names, k = 1, ):
	model = DecisionTreeClassifier( criterion = 'gini', max_depth = 12 )
	rfe = RFE( model, n_features_to_select = k )

	rfe = rfe.fit( x, y )

	ranking = rfe.ranking_
	selected = rfe.support_
	ranking = np.vstack( (ranking, selected) )
	(rows, cols) = ranking.shape

	rfe_selected = pd.DataFrame(
			data = ranking,
			index = np.array( range( 1, rows + 1 ) ),
			columns = np.array( range( 1, cols + 1 ) ) )
	rfe_selected.columns = names

	rfe_selected = rfe_selected.T # transpose
	rfe_selected.columns = [ 'rank', 'selected' ]
	return rfe_selected.sort_values( by = [ 'rank' ] )


def select_k_best( x, y, names, k = 1 ):
	test = SelectKBest( score_func = f_classif, k = k )
	fit = test.fit( x, y )

	ranking = np.vstack( ( fit.scores_, fit.pvalues_ ) )
	(rows, cols) = ranking.shape

	selected = pd.DataFrame(
		data = ranking,
		index = np.array( range( 1, rows + 1 ) ),
		columns = names )

	selected = selected.T
	selected.columns = [ 'score', 'p-value' ]

	return selected.sort_values( by = [ 'score' ], ascending = False )