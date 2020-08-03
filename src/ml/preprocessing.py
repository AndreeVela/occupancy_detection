import pandas as pd
from functools import reduce
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


def read_sheets( file_name, sheet_names ):

	sheets = pd.read_excel( file_name, sheet_name = sheet_names )

	for k, data in sheets.items():
		temp = data.drop( data.columns[ 10: ], axis = 1 )

		# megin together date and time columns

		temp[ 'Date' ] = temp[ 'Date' ].astype( 'str' ) + ' ' +  temp[ 'Hour' ].astype( 'str' )
		temp[ 'date' ] = pd.to_datetime( temp[ 'Date' ] )
		temp[ 'occ' ] = temp[ 'Personas' ].replace( { 0: 'E', 1:'L', 2:'L', 3:'M', 4:'M', 5:'M', 6:'H', 7:'H' } )
		temp = temp.drop( [ 'Date', 'Hour', 'Time Zone', 'Day', 'Personas' ], axis = 1 )

		# moving the timezone from UTC to CDT and removing timezone information

		temp = ( temp.set_index( 'date', drop = True )
				  .tz_localize( 'UTC' )
				  .tz_convert( 'US/Central' )
				  .tz_localize( None ) )

		# renaming columns

		rename_map = {
			'Pressure (hPa)': 'pre',
			'Altitude (m)': 'alt',
			'Humidity (%)': 'hum',
			'Temperature (C) ': 'tem',
			'Ventilador': 'ven' }
		sheets[ k ] = temp.rename( columns = rename_map )

	return reduce( lambda df1, df2: df1.append( df2 ), sheets.values() )


def pearson_corr( x, y, **kws ):
	( r, p ) = pearsonr( x, y )
	ax = plt.gca()
	ax.annotate( "r = {:.2f} ".format( r ),
				xy = (.1, .9), xycoords = ax.transAxes )


def make_patch_spines_invisible( ax ):
	ax.set_frame_on( True )
	ax.patch.set_visible( False )
	for sp in ax.spines.values():
		sp.set_visible( False )


def plot_env_vars( x, temp, hum, occ, title = 'Temperature, Humidity and Occupancy' ):
	fig, host = plt.subplots( 1, 1, figsize = ( 20, 6 ) )

	par1 = host.twinx()
	par2 = host.twinx()

	par2.spines[ 'right' ].set_position( ( 'axes', 1.04 ) )
	make_patch_spines_invisible( par2 )
	par2.spines[ 'right' ].set_visible( True )

	g1,  = host.plot( x, temp, 'royalblue', label = 'Temp' )
	g2,  = par1.plot( x, hum, 'forestgreen', label = 'Hum' )
	g3,  = par2.plot( x, occ.replace( { 'E':0, 'L': 1, 'M':2, 'H':3 } ), 'lightcoral', label = 'Occ' )

	host.set_xlabel( 'Date' )
	host.set_ylabel( 'Temp')
	par1.set_ylabel( 'Hum' )
	par2.set_ylabel( 'Occ' )

	par2.set_ylim( 0, 10 )
	par2.yaxis.set_major_locator( plt.IndexLocator( base = 1, offset = 0 ) )
	host.xaxis.set_major_locator( plt.AutoLocator() )

	plots = [ g1, g2, g3 ]
	host.legend( plots, [ l.get_label() for l in plots ] )

	fig.autofmt_xdate()
	plt.title( title )
	plt.show()


def plot_single( x, y, name ):
	fig, ax = plt.subplots( 1, 1, figsize = ( 20, 6 ) )

	g, = plt.plot( x, y, 'royalblue', label = name )

	ax.xaxis.set_major_locator( plt.AutoLocator() )
	ax.legend( [ g ], [ g.get_label() ] )
	ax.set_ylabel( name.capitalize() )

	fig.autofmt_xdate()
	plt.title( name.capitalize(), fontsize = 16 )
	plt.show()


def plot_svm( x, y, model ):
	fig, ax = plt.subplots( figsize = ( 12, 7 ) )

	# Removing to and right border

	ax.spines[ 'top' ].set_visible( False )
	ax.spines[ 'left' ].set_visible( False )
	ax.spines[ 'right' ].set_visible( False )

	# Create grid to evaluate model

	xx = np.linspace( -1, max( x ) + 1, len( x ) )
	yy = np.linspace( 0, max( y ) + 1, len( y ) )
	YY, XX = np.meshgrid( yy, xx )
	xy = np.vstack( [ XX.ravel(), YY.ravel() ] ).T

	# Assigning different colors to the classes

	colors = y
	colors = np.where( colors == 1, '#8C7298', '#4786D1' )

	# Plot the dataset

	ax.scatter( x, y, c = colors )

	# Get the separating hyperplane

	Z = model.decision_function( xy ).reshape( XX.shape )

	# Draw the decision boundary and margins

	ax.contour( XX, YY, Z, colors = 'k', levels = [ -1, 0, 1 ],
		alpha = 0.5, linestyles = [ '--', '-', '--' ] )

	# Highlight support vectors with a circle around them

	ax.scatter( model.support_vectors_[ :, 0 ], model.support_vectors_[ :, 1 ],
		s = 100, linewidth = 1, facecolors = 'none', edgecolors = 'k' )

	plt.show()