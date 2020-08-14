import numpy as np
import pandas as pd
from functools import reduce
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
from collections import Counter
from ml.constants import *



def read_sheets( file_name, sheet_names ):

	sheets = pd.read_excel( file_name, sheet_name = sheet_names )

	for k, data in sheets.items():
		temp = data.drop( data.columns[ 10: ], axis = 1 )

		# megin together date and time columns

		temp[ 'Date' ] = temp[ 'Date' ].astype( 'str' ) + ' ' +  temp[ 'Hour' ].astype( 'str' )
		temp[ 'date' ] = pd.to_datetime( temp[ 'Date' ] )
		temp[ 'occ' ] = temp[ 'Personas' ].replace( { 0: 'E', 1:'L', 2:'L', 3:'M', 4:'M', 5:'M', 6:'H', 7:'H' } )
		temp = temp.drop( [ 'Date', 'Hour', 'Time Zone', 'Day', 'Personas', 'Altitude (m)' ], axis = 1 )

		# moving the timezone from UTC to CDT and removing timezone information

		temp = ( temp.set_index( 'date', drop = True )
				  .tz_localize( 'UTC' )
				  .tz_convert( 'US/Central' )
				  .tz_localize( None ) )

		# renaming columns

		rename_map = {
			'Pressure (hPa)': 'pre',
			# 'Altitude (m)': 'alt',
			'Humidity (%)': 'hum',
			'Temperature (C) ': 'tem',
			'Ventilador': 'ven' }
		sheets[ k ] = temp.rename( columns = rename_map )

	return reduce( lambda df1, df2: df1.append( df2 ), sheets.values() )


# Methods to generate datasets

def resample_df( df, freq ):
	return ( df.resample( freq )
		.first()
		.dropna( axis = 0, how = 'any' ) )


def resample_df_avg( df, freq, agg = 'mean' ):

	return ( df.resample( freq )
		.agg( { 'pre': agg, 'hum': agg, 'tem': agg,
			'ven': mode, 'occ': mode, } )
		.dropna( axis = 0, how = 'any' ) )


def mode( series ):
	if( len( series ) ) :
		return series.mode()[ 0 ]


# Preprocessing data


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
