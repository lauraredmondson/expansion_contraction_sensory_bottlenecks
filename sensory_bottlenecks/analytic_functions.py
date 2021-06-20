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

# Analytic functions
import numpy as np  
import math

def bottleneck_allocation_2D(size, dens_ratio, act_ratio, sigma):
    '''
    Calculate allocation for 2D regions.

    Parameters
    ----------
    size : int
        Size of region 1
    dens_ratio : int
        dens_ratio of high to low density
    act_ratio : int
        variance dens_ratio for low : high
    sigma : float
        sigma value

    Returns
    -------
    r1_allo_x : ndarray
        xaxis points for region 1
    r1_allo_y : ndarray
        allocation for the region 1

    '''
    
    # region 1 values- l**2+m**2
    r1_vals = np.linspace(1**2+1**2, size**2+size**2, size*dens_ratio)
    
    # ordering for region 1
    order_r1 = np.zeros((len(r1_vals)))    
    
    for idx, val in enumerate(r1_vals):
        if size/np.sqrt(val) >= 1:
            order_r1[idx] = (np.pi*val)/4
        else:
            order_r1[idx] = (np.pi*val)/4 - (val*np.arccos(size/np.sqrt(val))
                                                         - size*np.sqrt(val-size**2))

    # region 2 values
    r2_vals = (r1_vals*np.sqrt(dens_ratio)*act_ratio) + (((size**2*sigma**2*np.sqrt(dens_ratio)*act_ratio)
                                                                -(size**2*sigma**2))/ (np.pi**2))
    
    r2_vals[r2_vals < 0] = 0 # for plotting
    
    # calculate ordering for high allocation
    order_r2 = np.zeros((len(r2_vals))) 
    
    for idx, val in enumerate(r2_vals):
        if math.isclose(order_r2[idx-1],(size*np.sqrt(dens_ratio))**2, rel_tol=.00003): # idx > 0 and 
            order_r2[idx] = (size*np.sqrt(dens_ratio))**2 # makes sure only includes valid points
        elif (size*np.sqrt(dens_ratio))/np.sqrt(val) >= 1:
            order_r2[idx] = (np.pi*val)/4
        else:
            order_r2[idx] = (np.pi*val)/4 - (val*np.arccos((size*np.sqrt(dens_ratio))
                                                          /np.sqrt(val)) - (size*np.sqrt(dens_ratio))
                                                          *np.sqrt(val-(size*np.sqrt(dens_ratio))**2))

    # total points
    r1_allo = order_r2 + order_r1 
    
    # Plotting
    # x axis points convert to percentage
    r1_allo_x = r1_allo / ((np.sqrt(dens_ratio)*size)**2 + size**2)*100
    
    # y axis points convert to percentage
    r1_allo_y = order_r1/ r1_allo * 100
    
    # for filled colour plots
    r1_allo_x = np.insert(r1_allo_x, 0, r1_allo_x[0])
    r1_allo_y = np.insert(r1_allo_y, 0, 0)
    
    # calculate expected decay after run out of eigenvalues
    r1_allo_x, r1_allo_y = curve_plot_calc(r1_allo_x, r1_allo_y, dens_ratio**2)
    
    return r1_allo_x, r1_allo_y


def bottleneck_allocation_1D(size, dens_ratio, act_ratio, sigma):
    '''
    Calculate allocation for 1D regions.

    Parameters
    ----------
    size : int
        Size of region 1
    dens_ratio : int
        density ratio value
    act_ratio : int
        activation ratio value
    sigma : float
        sigma value

    Returns
    -------
    r1_allo_x : ndarray
        xaxis points for region 1
    r1_allo_y : ndarray
        allocation for the region 1

    '''
    # region 1 points
    r1 = np.linspace(1,size,size)
    
    # calculate number allocated to h for each l, based on Eq11
    r2_allo = np.sqrt((r1**2*np.pi**2*dens_ratio*act_ratio) + (size**2*sigma**2*dens_ratio*act_ratio)
                                                                       - (size**2*sigma**2)) / (np.pi)  
    
    # for plotting
    r2_allo[np.isnan(r2_allo)] = 0
    
    # include only valid values
    index = np.max(np.where(r2_allo <= dens_ratio*size))
    r2_allo = r2_allo[:index+1]

    # r1 allocation
    r1_allo = np.linspace(1, len(r2_allo), len(r2_allo))
    
    # total number assigned for percentage calculation
    total_assigned = r2_allo + r1_allo
        
    # calculate the x value of each allocation point as percentage
    r1_allo_x = total_assigned/((dens_ratio+1)*size)*100
    
    # calculate the y value of each allocation point as percentage
    r1_allo_y = r1_allo / total_assigned *100
    
    # for filled colour plots
    r1_allo_x = np.insert(r1_allo_x,0, r1_allo_x[0])
    r1_allo_y = np.insert(r1_allo_y, 0, 0)
    
    # calculate decay after run out of eigenvalues
    r1_allo_x, r1_allo_y = curve_plot_calc(r1_allo_x, r1_allo_y, dens_ratio)
    
    return r1_allo_x, r1_allo_y


def curve_plot_calc(allo_x, allo_y, dens_ratio):
    '''
    For plotting. Calculate curve from limit to proportional
    density.

    Parameters
    ----------
    allo_x : ndarray
        x axis values for allocation of region 1
    allo_y : ndarray
        allocation of region 1 
    dens_ratio : int
        densiy ratio value

    Returns
    -------
    allo_x : ndarray
        x axis values for allocation of region 1
    allo_y : ndarray
        allocation of region 1 
    '''
    
    if allo_y[-1]  > 100/ (dens_ratio+1): # if allocation is above density ratio

        values = np.linspace(allo_x[-1], 100, 100) #int(100-allo_x[-1])*10)
        allo_x, allo_y = np.append(allo_x, values), np.append(allo_y, (allo_y[-1] * allo_x[-1])/values)

    elif allo_y[-1]  < 100/ (dens_ratio+1): # if allocation is below density ratio

        values = np.linspace(allo_x[-1], 100, 100) #int(100-allo_x[-1])*10)
        allo_x, allo_y = np.append(allo_x, values), 100-np.append(100-allo_y, ((100-allo_y)[-1] * allo_x[-1])/values)
            
    return allo_x, allo_y
 
