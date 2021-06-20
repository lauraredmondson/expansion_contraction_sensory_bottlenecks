"""
Expansion and contraction of resource allocation in sensory bottlenecks.
Edmondson, L, R., Jiménez Rodríguez, A., Saal, H. P.

Written in 2021 by Laura R. Edmondson.
To the extent possible under law, the author(s) have dedicated all copyright 
and related and neighboring rights to this software to the public domain 
worldwide. This software is distributed without any warranty.
You should have received a copy of the CC0 Public Domain Dedication 
along with this software. 
If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
"""


import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib as mpl

def snm_ray_plot(data, color, title):
    """
    Plots information about ray eg. raw innervation density values
    or usage values

    Parameters
    ----------
    data : ndarray (size: 11)
        Fill data values for each ray eg. density or usage
    color : str
        Name of matplotlib colormap for plotting.
        In paper,'Greens' for usage, and 'Purples' for desities.

    Returns
    -------
    None.

    """
    
    with open('sensory_bottlenecks/snm_plotting_params.pk', 'rb') as handle:  # load ray outline data
        ray_data = pk.load(handle)

    data = data.astype('int') # convert data for plotting
    fill_data = np.round(data-np.min(data)).astype('int') # calculate min-max value range for colour map
    fcol = cm.get_cmap(color, max(fill_data)) # create colormap
    
    fig, axs = plt.subplots()
    
    # plot each ray and fill with color corresponding to data
    for idx, ray in enumerate(ray_data): 
        plt.plot(ray_data[ray][:,0], ray_data[ray][:,1], 'k')
        axs.fill(ray_data[ray][:,0], ray_data[ray][:,1], alpha=1, fc=fcol(fill_data[idx]), ec='none')
    
    plt.axis('equal')
    plt.axis('off')
    plt.title(title)
    
    # create colorbar
    sm = plt.cm.ScalarMappable(cmap=fcol, norm=mpl.colors.Normalize(vmin=min(data), vmax=max(data)))
    cbar = plt.colorbar(sm, ticks=np.linspace(min(data), max(data),10)) # add cbar
    cbar.ax.set_yticklabels(np.round(np.linspace(min(data), max(data),10)).astype('int')) # add cbar labels


def snm_scatter_ray(each_ray_allo, cortex_per):
    """
    Plots predicted versus empirical allocations

    Parameters
    ----------
    each_ray_allo : ndarray (size: 11)
        predicted allocations for each ray
    cortex_per : ndarray (size: 11)
        empirical allocations for each ray

    Returns
    -------
    None.

    """
    
    fcol = cm.get_cmap('plasma', 13)
    
    fig, axs = plt.subplots(1,1, figsize=(3,3))
    
    for i in range(len(each_ray_allo)):
        axs.scatter(each_ray_allo[i], cortex_per[i], color=fcol(i))
        
    axs.plot([0,50],[0,50],'lightgrey',linestyle='--') # identity line
    axs.set_title('Full model')
    axs.set_xlim([0,30])
    axs.set_ylim([0,30])
    
    axs.set(xlabel='Model allocation %')
    axs.set(ylabel='Cortical allocation %')
    axs.set_aspect('equal')
    
    axs.set_xticks(np.linspace(0,30,7))
    axs.set_xticklabels(np.linspace(0,30,7).astype('int'))
    axs.set_yticks(np.linspace(0,30,7))
    axs.set_yticklabels(np.linspace(0,30,7).astype('int'))


def snm_bar_plot(rmses):
    """
    Bar plot for the three models RMSE values.

    Parameters
    ----------
    rmses : list (size: 3)
        rmse values for the three models: density only, usage only, full model

    Returns
    -------
    None.

    """
    
    colours = ['#65479c', '#2ca249', '#b52e8e'] # color codes for each model
    
    plt.figure(figsize=(4,3))
    
    for idx, rmse in enumerate(rmses):
        plt.bar(idx, rmse, color=colours[idx])
        
    plt.ylabel('RMSE')
    plt.xticks(np.arange(3), labels=['Densities \n only', 'Receptor usage \n only',  'Full model'])
    plt.ylim([0,np.max(rmses)+0.25])


