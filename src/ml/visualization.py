import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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