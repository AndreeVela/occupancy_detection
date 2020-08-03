import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
from collections import Counter
from ml.constants import *


def agg_date( col ):
	return col.iloc[ 0 ] # left labeling


def agg_occupancy( col ):
	return col.mode()[ 0 ]


def groupby_10( g, agg ):
	return g.groupby( np.arange( len( g ) ) // 10 ).agg( {
		'date': agg_date,
		'occ': agg_occupancy,
		'pre': agg,
		'alt': agg,
		'hum': agg,
		'tem': agg } )


def df_10sec_avg( df, agg ):
	temp =  ( df.groupby( df.date.dt.floor( 'D' ) )
		.apply( groupby_10, agg )
		.reset_index( drop = True ) )

	temp.columns = [ c[ 0 ] for c in temp.columns ]
	temp = ( temp
		.rename( columns = { 'Date_agg_date': 'date', 'Occ_agg_occupancy': 'occ' } )
		.sort_values( by = 'date' )
		.reset_index( drop = True ) )

	return temp


def df_1min_samples( df ):
	return ( df.resample( '1T' )
		.first()
		.dropna( axis = 0, how = 'any' ) )


def df_5min_samples( df ):
	return ( df.resample( '5T' )
		.first()
		.dropna( axis = 0, how = 'any' ) )


def split_data( df, test_size = 0.20 ):
	x_train, x_test, y_train, y_test = train_test_split(
		df.drop( [ 'occ' ], axis = 1 ),
		df.occ,
		test_size = .20,
		random_state = 0 )

	return x_train, x_test, y_train, y_test


def standardize( x_train, x_test ):
	scaler = StandardScaler()
	scaler.fit( x_train )

	x_train = scaler.transform( x_train )
	x_test = scaler.transform( x_test )

	return x_train, x_test


def balance_df( x_train, y_train, neighborgs ):
	oversampler = ADASYN(
		sampling_strategy = 'not majority',
		n_neighbors = neighborgs,
		random_state = 42 )

	x_train, y_train = oversampler.fit_resample( x_train, y_train )

	return x_train, y_train

