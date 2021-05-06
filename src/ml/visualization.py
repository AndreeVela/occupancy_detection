import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from ml.gini import gini




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

        
def plot_env_vars2( x, temp, hum, occ, figsize = ( 20, 5 ), df = None ):
    sns.set_theme( style = 'white' )
    
    colors = { 'temp': 'royalblue', 'hum': 'forestgreen', 'occ': 'lightcoral' }
    fig = plt.figure( figsize = figsize )
    gs = gridspec.GridSpec( 6, 1 )
     
    # creting twin spines and setting up the plot frame     
    
    host = plt.subplot( gs[ :5, : ] )
    host.spines[ 'top' ].set_visible( False )
    host.spines[ 'bottom' ].set_visible( False )
    host.spines[ 'left' ].set_color( colors[ 'temp' ] )
    
    par1 = host.twinx()
    make_patch_spines_invisible( par1 )
    par1.spines[ 'right' ].set_visible( True )
    par1.spines[ 'right' ].set_color( colors[ 'hum' ] )
    
    # plotting data
    
    index = list( range( len( x ) ) )
    g1,  = host.plot( index, temp, 'royalblue', label = 'Temp' )
    g2,  = par1.plot( index, hum, 'forestgreen', label = 'Hum' )
    
    # labels and tick labels settings
    
    host.set_ylabel( 'Temp' )
    host.tick_params( axis = 'y', color = colors[ 'temp' ] ) 
    host.tick_params( axis = 'x', which = 'both', bottom = False, labelbottom = False )
    par1.set_ylabel( 'Hum' )
    par1.tick_params( axis = 'y', color = colors[ 'hum' ] )
    # plt.setp( host.get_xticklabels(), visible = False )  
    
    # occupancy plot
    
    host2 = plt.subplot( gs[ 5:, : ] )
    host2.spines[ 'top' ].set_visible( False )
    host2.spines[ 'right' ].set_visible( False )
    host2.spines[ 'left' ].set_color( colors[ 'occ' ] )
    
    g3,  = host2.plot( index, occ.replace( { 'E':0, 'L': 1, 'M':2, 'H':3 } ), 'lightcoral', label = 'Occ' )
    
    host2.set_xlabel( 'Date' )
    
    host2.set_ylabel( 'Occ' )
    host2.set_yticks( [ 0, 1, 2, 3 ] )
    host2.set_yticklabels( labels = [ 'E', 'L', 'M', 'H' ] )
    host2.tick_params( axis = 'y', which = 'both', left = True, labelleft = True, color = colors[ 'occ' ] )
    host2.tick_params( axis = 'x', which = 'both', bottom = True )
    host2.set_ylim( -1, 4 )
    
    # days = mdates.DayLocator()
    # host.xaxis.set_major_locator( days )
    
    # Obtaining ticks positions. 
    # Each tick signals the end of the day.
    
    temp = df.groupby( pd.Grouper( level = 'date', freq = 'D' ) ).first().dropna()
    ticks = [ 0 ]
    for i in range( temp.shape[ 0 ] ):
        day = temp.index[ i ] + pd.to_timedelta( 1, unit = 'd' )
        stop = df[ df.index < day ].iloc[ - 1 ].name
        loc = df.index.get_loc( stop )
        ticks.append( loc )
    host2.set_xticks( ticks )
    host2.set_xticklabels( labels = x[ ticks ].strftime( '%Y-%m-%d %H:%M:%S' ).astype( str ), ha = 'right', rotation = 45 )
                        
    plots = [ g1, g2, g3 ]
    host.legend( plots, [ l.get_label() for l in plots ] )

    fig.autofmt_xdate()
    plt.show()
    
    return fig


def plot_objects( df, colors, hue_order, figsize = ( 10, 10 ) ):
    plt.rcParams[ "axes.grid.axis" ] = 'y'
    plt.rcParams[ "axes.grid" ] = True
    
    fig = plt.figure( figsize = figsize )
    gs = gridspec.GridSpec( 1, 18 )
    width = 0.35
    
    # Counting by date

#     temp = ( df[ [ 'pre', 'occ'] ]
#         .groupby( [ pd.Grouper( level = 'date', freq = 'D' ), 'occ' ] )
#         .count()
#         .unstack( level = 'occ', fill_value = 0 ) )
#     temp.columns = temp.columns.droplevel( 0 )

#     labels = temp.index.strftime( '%Y-%m-%d' )
#     bottom = [ 0 ] * temp.shape[ 0 ]

#     ax1 = plt.subplot( gs[ :, :13 ] )
#     for level in hue_order:
#         ax1.bar( labels, temp[ level ], width, label = level, color = colors [ level ], bottom = bottom   )
#         bottom +=  temp[ level ]
    
#     ax1.set_xlabel( 'Date' )
#     ax1.set_ylabel( 'Count' )
#     ax1.spines[ 'top' ].set_visible( False )
#     ax1.spines[ 'right' ].set_visible( False )
#     ax1.tick_params( axis = 'x', which = 'both', bottom = True )
#     ax1.tick_params( axis = 'y', which = 'both', left = True )
#     #     ax1.set_xticks( temp.index.unique().strftime( '%Y-%m-%d' ) )
#     #     ax1.set_xticklabels( labels = temp.index.strftime( '%Y-%m-%d' ), ha = 'center' ) # rotation = 45,
#     plt.legend( loc = 'upper right', title = 'Occupancy' )

    # Counting by occupancy level

    temp = df.groupby( 'occ' ).count()
    ax2 = plt.subplot( gs[ :, : ], ) #sharey = ax1 
    plt.bar( hue_order, temp[ 'pre' ].loc[ hue_order ], width, color = [ colors[ k ] for k in hue_order ] ) 

    ax2.set_xlabel( 'Occupancy' )
    ax2.spines[ 'top' ].set_visible( False )
    ax2.spines[ 'right' ].set_visible( False )
    ax2.spines[ 'left' ].set_visible( False )
    ax2.tick_params( axis = 'y', which = 'both', labelleft = True, left = True )
    ax2.tick_params( axis = 'x', which = 'both', bottom = True )

    plt.show()
    
    return fig


def lorenz_plot( gini_score, data, var_name, color, ax, level = False, ):
    x = range( 1, 101 )
    y = [ np.percentile( data, i ) for i in x ]
    delta = ( data.max() - data.min() ) / 100
    y_line = [ delta * i for i in x ]

    ax.bar( x, y, color = color )
    ax.plot( x, y_line, color = 'gray', lw = 4,  linestyle = '--' )
    ax.set_xlabel( 'Percentile' )
    
    label = 'Cumulative Sum of ' + var_name
    label = label + ' for %s occ. level' % level if level else label
    
    ax.set_ylabel( label )
    ax.text( 50, ax.get_yticks()[ -2 ], 'Gini = %0.3f' % gini_score, fontsize = 14, fontweight = 'bold' )


def env_var_lorenz_plots( df, color, levels, var, var_name ):
    df_scores = df.groupby( 'occ' )[ var ].apply( lambda col: gini( np.array( col ) ) ).loc[ levels ]
    figures = []
    
    # plot by occ level
    
    for i, level in enumerate( levels ):
        fig, ax = plt.subplots( figsize = ( 8, 8 ) )
        data = df[ df.occ == level ].sort_values( [ var ] )[ var ].cumsum()
        
        lorenz_plot( df_scores[ level ], data, var_name, color, ax, level = level )
        figures.append( fig )
    
    # plot of complete dataset
    
    fig, ax = plt.subplots( figsize = ( 16, 8 ) )
    gini_score = gini( np.array( df[ var ] ) )
    data = df[ var ].sort_values().cumsum() 
    
    lorenz_plot( gini_score, data, var_name, color, ax )
    figures.append( fig )
    
    return figures


def plot_gini_by_day( df, var, var_name, order, colors, figsize, limit = False ):
    plt.rcParams[ "axes.grid.axis" ] = 'y'
    plt.rcParams[ "axes.grid" ] = True
    
    fig, ax = plt.subplots( figsize = figsize )
    g = sns.kdeplot( data = df, x = var, hue = 'occ', fill = True, palette = colors, hue_order = order, ax = ax )
    ax.set_xlabel( var_name )

    if limit: 
        ax.set_xlim( limit )
    
    return fig
    

def plot_single( x, y, name, figsize = ( 20, 6 ) ):
	fig, ax = plt.subplots( 1, 1, figsize = figsize )

	g, = plt.plot( x, y, 'royalblue', label = name )

	ax.xaxis.set_major_locator( plt.AutoLocator() )
	ax.legend( [ g ], [ g.get_label() ] )
	ax.set_ylabel( name.capitalize() )

	fig.autofmt_xdate()
	plt.title( name.capitalize(), fontsize = 16 )
	plt.show()


def plot_svm( x, y, model, figsize = ( 12, 7 ) ):
	fig, ax = plt.subplots( figsize = figsize )

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