# Plotting functions for sensory bottlenecks
import matplotlib.pyplot as plt
import numpy as np

def allocation_plot(allo_x, allo_y, dens_ratio, act_ratio, plot_type='1D'):
    """
    Plots allocation over all bottleneck sizes. Includes limit and proportional
    density lines.

    Parameters
    ----------
    allo_x : ndarray
        X values for plotting allocation
        
    allo_y : ndarray
        Allocation for region 1
        
    dens_ratio : int
        ratio for the density
        
    act_ratio : int
        ratio for the activation
        
    plot_type : str, optional
        Whether data for 1D or 2D sim. The default is '1D'.

    Returns
    -------
    None.

    """
    
    plt.figure()

    if plot_type == '1D':
        v_line = 1/(1+np.sqrt(dens_ratio*act_ratio))*100
        plt.plot([0,100],[v_line,v_line],linestyle='--',color='#d1d1d1',Label = r'$\frac{1}{1 + \sqrt{ad}}$')
        
    elif plot_type == '2D':
        v_line = 1/(1+np.sqrt(dens_ratio)*act_ratio)*100
        plt.plot([0,100],[v_line,v_line],linestyle='--',color='#d1d1d1',Label = r'$\frac{1}{1 + a \sqrt{d}}$')
        
    # calculate density line and add
    d_line = 100/(dens_ratio+1)
    plt.plot([0,100],[d_line,d_line],linestyle='--',color='#9c2c2c',label='Proportional density')
        
    # plot allocation line data
    plt.plot(allo_x, allo_y)
        
    # add title, axis and legend
    plt.xticks(np.linspace(0,100,11))
    plt.xlim([0,100])
    plt.ylim([0,100])
    plt.ylabel('Allocation [%]')
    plt.xlabel('Bottleneck width [%]')
    plt.legend()
    plt.title('Allocation for density:{}, activation:{}'.format(dens_ratio, act_ratio))


    
    
    
    
    
    
    
    
    
    
    
    