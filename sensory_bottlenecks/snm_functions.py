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

# Functions for SNM
import numpy as np  
from sklearn.metrics import mean_squared_error

def snm_sim(size, dens_ratios, act_ratios, sigmas):
    '''
    Runs the star-nosed mole simulation.
    Calculates the eigenvalues for each region.
    Returns allocation percentages
    
    Parameters
    ----------
    size : ndarray (size: 11)
        size of each ray along one dimension
    dens_ratios : ndarray (size: 11)
        density of receptors on each ray.
    act_ratios : ndarray (size: 11)
        activation for each ray
    sigmas :  ndarray (size: 11)
        sigmas for the exponential function of each ray

    Returns
    -------
    ndarray, (size:(total no.eigenvalues for all rays, 11))
        percentage allocations for each ray (one column each ray).
    '''
     
    all_eigs = [] # list all eigenvalues
    ray_idx = [] # list all ray indexes
    
    # assign one ray to be base region
    idx = np.argmax(dens_ratios)

    # loop through rays
    for i in range(11):
        
        # calculate eigenvalues
        grid_vals = np.linspace(1, int(np.round(size[i]*dens_ratios[i])), int(np.round(size[i]*dens_ratios[i])))
        xv, yv = np.meshgrid(grid_vals, grid_vals)
        
        x1 = np.reshape(xv,-1)
        x2 = np.reshape(yv,-1) 
        
        # calculate density with respect to base region
        ray_density = dens_ratios[idx] / dens_ratios[i]
        
        eigenvals = (2*sigmas[i]*act_ratios[i])/((x1**2*np.pi**2*size[i]**(-2)*ray_density)+
                                             (x2**2*np.pi**2*size[i]**(-2)*ray_density)+
                                             (sigmas[i]**2*ray_density))
        
        all_eigs.append(eigenvals)
        ray_idx.append(np.tile(i,len(eigenvals))) 
       
    # add eigenvalue and ray index data to same array
    all_eigs = np.vstack([np.concatenate(all_eigs, axis=0),np.concatenate(ray_idx, axis=0)]).T
    
    return snm_calc_ordering(all_eigs)  # from calculate allocation/ ordering


def snm_calc_ordering(all_eigs):
    '''
    Calculates the ordering of each ray, and returns percentage allocation
    for each ray at each bottleneck size.

    Parameters
    ----------
    all_eigs : ndarray, (size:(total no.eigenvalues for all rays, 11))
        Percentage allocations for each ray (one column each ray).

    Returns
    -------
    ndarray, (size:(total no.eigenvalues for all rays, 11))
        Percentage allocations for each ray at each bottleneck size.

    '''

    # sort the data in descending order based on first column
    sort_rays = all_eigs[all_eigs[:,0].argsort()][::-1]
    
    # matrix for running total for each array
    all_ray_data = np.zeros((len(sort_rays),11))
    
    # calculate each ray's allocation for each bottleneck width
    for i in range(len(sort_rays)):
        all_ray_data[i,int(sort_rays[i,1])] = 100/len(sort_rays)
        
    # cumulative sum of this array
    ray_allo = np.cumsum(all_ray_data,0)
        
    # calculate allocations at each bottleneck size as percentage
    return ray_allo/np.sum(ray_allo,1)[:,None]*100


def snm_rmse_all_bottlenecks(ray_allo_per, cortex_per):
    """
    Calculates the RMSE between empirical cortical data and all
    bottleneck allocations.

    Parameters
    ----------
    ray_allo_per : ndarray (size: 11)
        predicted allocations for each ray
    cortex_per : ndarray (size: 11)
        empirical allocations for each ray

    Returns
    -------
    rmse_allo : ndarray (size: 11)
        predicted ray allocations at best fitting bottleneck
    best_rmse_val : float
        rmse for best fitting bottleneck

    """
    
    # calculate one percent of total
    one_per = int((len(ray_allo_per)/100))
    
    #preallocate arrays
    rmse = np.zeros((len(ray_allo_per)))

    for i in range(len(ray_allo_per)): # calculate error for each ray at each bottleneck size
        rmse[i] = mean_squared_error(cortex_per, ray_allo_per[i,:], squared=False)  # rmse

    # find best fitting index
    best_fit_rmse = np.argmin(rmse[one_per:np.size(ray_allo_per,0)]) + one_per # best fit- min rmse

    rmse_allo = ray_allo_per[best_fit_rmse,:]
    
    best_rmse_val = rmse[best_fit_rmse]

    return rmse_allo, best_rmse_val

