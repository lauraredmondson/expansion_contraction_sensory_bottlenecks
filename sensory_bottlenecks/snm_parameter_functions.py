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

# extract SNM rays
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import random
from shapely.geometry import Point, Polygon, LinearRing
from shapely import geometry
import math
from itertools import repeat
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import curve_fit

# %% Functions for rectangle calc
def rotate(origin, point, angle):
    """
    Rotate point by angle around origin.

    Parameters
    ----------
    origin : tuple
        coordinates of origin
    point : ndarray (size: 2,2)
        coordinates of point
    angle : float
        angle to rotate in radians

    Returns
    -------
    float
        rotated x coordinate
    float
        rotated y coordinate

    """
    rot_x = origin[0] + math.cos(angle) * (point[0] - origin[0]) - math.sin(angle) * (point[1] - origin[1])
    rot_y = origin[1] + math.sin(angle) * (point[0] - origin[0]) + math.cos(angle) * (point[1] - origin[1])
    
    return rot_x, rot_y


def distance(p1,p2):
    """
    Euclidean distance between two points.

    Parameters
    ----------
    p1 : ndarray (size: 2)
        coordinate point 1
    p2 : ndarray (size: 2)
        coordinate point 2

    Returns
    -------
    float
        Euclidean distance

    """
    return np.linalg.norm(p1-p2)


def rotate_rays(ray_data):
    """
    Rotate the rays for SNM such that each fits into smaller bounding box.

    Parameters
    ----------
    ray_data : dict
        region data for each ray.

    Returns
    -------
    ray_data_rot : dict
        region data for each ray rotated to fit within bounding box.

    """
    ray_data_rot = {}
    
    rays = ['Ray01','Ray02','Ray03','Ray04','Ray05','Ray06','Ray07','Ray08','Ray09','Ray10','Ray11']

    # add data to dictionary
    for i in range(len(rays)):
        # distance between all coords:
        boundaries_new = ray_data[rays[i]]['boundary_center'] 
        chord_dist = np.triu(np.array([[ np.linalg.norm(i-j) for j in boundaries_new] for i in boundaries_new]))
        
        ind = np.unravel_index(np.argmax(chord_dist, axis=None), chord_dist.shape)
        coords = np.array(([boundaries_new[ind[0],:], boundaries_new[ind[1],:]]))
        coords = np.vstack([np.array(([0,0])), coords])
        
        # calc dist from zero
        dist_zero = np.argmin([distance(*pair) for pair in zip(repeat( coords[0]), coords[1:])]) + 1
        
        # center on zero
        boundaries_new = boundaries_new - np.tile(coords[dist_zero,:], [len(boundaries_new),1])
        
        # align vector
        if dist_zero == 1:
            vec1 = np.array((0,10))
            vec2 = coords[2,:] - coords[dist_zero,:]
        else:
            vec1 = coords[1,:] - coords[dist_zero,:]
            vec2 = np.array((0,10))
            
        #calculate bounding box of these re-centered coords
        unit_vector1 = vec1 / np.linalg.norm(vec1)
        unit_vector2 = vec2 / np.linalg.norm(vec2)
        dot_product = np.dot(unit_vector1, unit_vector2)
        angle = np.arccos(dot_product) #angle in radian
        
        # view rotated
        bound_rot = np.zeros((len(boundaries_new),2))
        origin = (0,0)
        for j in range(len(bound_rot)):
            bound_rot[j,:] = rotate(origin, boundaries_new[j,:], angle)
        
        # calculate size as a rectangle bbox in each direction
        bbox_min_x, bbox_max_x = np.min(bound_rot[:,0]), np.max(bound_rot[:,0])
        bbox_min_y, bbox_max_y = np.min(bound_rot[:,1]), np.max(bound_rot[:,1])
        
        width = np.max(bound_rot[:,0]) - np.min(bound_rot[:,0])
        height = np.max(bound_rot[:,1]) - np.min(bound_rot[:,1])
        
        ray_data_rot[rays[i]] = {}
        ray_data_rot[rays[i]]['boundary'] = bound_rot
        ray_data_rot[rays[i]]['bbox_x'] = np.array(([bbox_min_x,bbox_max_x]))
        ray_data_rot[rays[i]]['bbox_y'] = np.array(([bbox_min_y,bbox_max_y]))
        ray_data_rot[rays[i]]['dimensions_w'] = width
        ray_data_rot[rays[i]]['dimensions_h'] = height
        
#    with open("snm_rotated_ray_data", "wb") as input_file:
#        pk.dump(ray_data_rot,input_file)   
        
    return ray_data_rot

    
# %% data dictionary
def SNM_ray_data_dict(plot_rays=False):
    """
    Load pre-computed ray data- coordinate boundaries of each ray extracted from
    image.
    
    Plots each ray if required.

    Parameters
    ----------
    plot_rays : bool, optional
        Plots the outline of the rays. The default is False.

    Returns
    -------
    ray_data : dict
        Data for each SNM ray- size, bounding box, and coordinates.

    """
    # sort the above data into a dictionary
    rays = ['Ray01','Ray02','Ray03','Ray04','Ray05','Ray06','Ray07','Ray08','Ray09','Ray10','Ray11']
    
    # load ray data
    with open("snm_region_data.pk", "rb") as input_file:
        ray_data = pk.load(input_file)
        
    # plot the rays
    if plot_rays:
        fig, axs = plt.subplots()
        
        for ray in rays:
            pointList = ray_data[ray]['boundary']
            poly = geometry.Polygon([[p[0], p[1]] for p in pointList])
            x,y = poly.exterior.xy
            plt.plot(x,y, 'k')
        
        plt.axis('off')
        plt.axis('equal')
    
    return ray_data


#%% buffer functions

def build_ray_buffers(plot=False):
    """
    Calculates points inside the ray and a buffer boundary for the placement of
    stimuli to avoid border effects.
    
    Saves buffer data.

    Parameters
    ----------
    plot : bool, optional
        whether to plot points in rays and buffer. The default is False.

    Returns
    -------
    None.

    """
    rays = ['Ray01','Ray02','Ray03','Ray04','Ray05','Ray06','Ray07','Ray08','Ray09','Ray10','Ray11']

    snm_rays = SNM_ray_data_dict() # load surface data
    
    ray_data_center = rotate_rays(snm_rays) # center and rotate ray data
    
    all_areas = np.zeros((len(rays)))
    
    for i in range(len(snm_rays)):
        ray_data_center[rays[i]]['bbox_min'] = np.array((ray_data_center[rays[i]]['bbox_x'][0], ray_data_center[rays[i]]['bbox_y'][0]))
        ray_data_center[rays[i]]['bbox_max'] = np.array((ray_data_center[rays[i]]['bbox_x'][1], ray_data_center[rays[i]]['bbox_y'][1]))
        ray_data_center[rays[i]]['area'] = snm_rays[rays[i]]['area'] 
        all_areas[i] = snm_rays[rays[i]]['area'] 
        
    # view all the points in rays and add to ray data
    for i in range(len(rays)):
        single_ray_data = ray_data_center[rays[i]]
        
        pointList = single_ray_data['boundary']
        poly = geometry.Polygon([[p[0], p[1]] for p in pointList])
        pw_x, pw_y = random_points_within(ray_data_center[rays[i]], 2000)
        idx = np.argsort(pw_x)
        points = np.vstack([pw_x,pw_y]).T
        points = points[idx,:]
        
        if plot: # plot rays with points
            x,y = poly.exterior.xy
            plt.figure()
            plt.plot(x,y,color = 'k')
            plt.scatter(pw_x,pw_y)
            plt.scatter(points[:,0],points[:,1],5,'red')
            plt.axis('equal')
            plt.title('Receptor tiling within the ray')
        
        ray_data_center[rays[i]]['points'] = points # add to ray data
        
    all_radius = np.array(([1,1.5,2,2.5,3,3.5,4,4.5]))/2
    
    for i in range(len(ray_data_center)): # find shape centroid
        
        buffer_bounds = {}
        current_ray = ray_data_center[rays[i]]
        coords = current_ray['boundary']
        
        # create buffer
        ring = LinearRing(coords)
        ring_poly = Polygon(ring)
        x_s, y_s = ring_poly.exterior.coords.xy
        
        if plot: # plot the buffers
            plt.figure() 
            plt.plot(x_s,y_s, color='k')     
            plt.axis('equal')
        
        for j in range(len(all_radius)):
            radius = all_radius[j] * 28 # scaling to be in same space as ray coords
            
            buffer_bounds[str(all_radius[j])] = {}
            buffer = Polygon(ring_poly.buffer(radius).exterior, [ring])
            coords_buffer = np.array((buffer.exterior.coords.xy)).T
            buffer_bounds[str(all_radius[j])]

            # bounding box
            buffer_bounds[str(all_radius[j])]['coords'] = coords_buffer
            buffer_bounds[str(all_radius[j])]['bbox'] = np.array(([np.min(coords_buffer[:,0]),np.max(coords_buffer[:,0])],
                                                                    [np.min(coords_buffer[:,1]),np.max(coords_buffer[:,1])]))
            # plot the buffer bounding box
            if plot:
                plt.plot(coords_buffer[:,0],coords_buffer[:,1])
                plt.scatter([buffer_bounds[str(all_radius[j])]['bbox'][0,0],buffer_bounds[str(all_radius[j])]['bbox'][0,1]],
                            [buffer_bounds[str(all_radius[j])]['bbox'][1,0],buffer_bounds[str(all_radius[j])]['bbox'][1,1]])
                plt.title('Buffer boundaries and bounding boxes for each stimuli size')
                
        ray_data_center[rays[i]]['buffer_bounds'] = buffer_bounds
    
    # save ray data center
    with open('snm_params_ray_center_buffer.pk', 'wb') as handle:
        pk.dump(ray_data_center, handle)
    

# %% contact/ stim dropping functions
def random_points_within(single_ray_data, num_points):
    """
    Create receptor positions within the ray.

    Parameters
    ----------
    single_ray_data : dict
        region data for a single ray.
    num_points : int
        number of receptors to tile into ray.

    Returns
    -------
    final_points_x : ndarray
        x coordinates for all points.
    final_points_y : ndarray
        y coordinates for all points.

   """
    # try grid of x size with spacing
    bbox_min, bbox_max  = single_ray_data['bbox_min'], single_ray_data['bbox_max']
   
    pointList = single_ray_data['boundary']
    polygon = geometry.Polygon([[p[0], p[1]] for p in pointList])
    line = geometry.LineString(list(polygon.exterior.coords))
   
    start_val, total = 30, 0
    start_vals, point_num_list = [], []
   
    while total < num_points:
       
       total_count = 0
       start_val -= 1
       
       xval = np.linspace(bbox_min[0],bbox_max[0],int((bbox_max[0] - bbox_min[0])/start_val))
       yval =np.linspace(bbox_min[1],bbox_max[1],int((bbox_max[1] - bbox_min[1])/start_val))
       
       x, y = np.meshgrid(xval, yval)
       x, y = x.ravel(), y.ravel()
       final_points_x, final_points_y = [], []
       
       # calculate number inside polygon
       for i in range(len(x)):
           point = Point(x[i], y[i])
    
           if polygon.contains(point) or line.contains(point):
                total_count += 1
                final_points_x.append(x[i])
                final_points_y.append(y[i])
                
       total = total_count
       point_num_list.append(total)
       start_vals.append(start_val)
       
    if np.abs(num_points-start_vals[-2]) < np.abs(num_points-start_vals[-1]):
       xval = np.linspace(bbox_min[0],bbox_max[0],int((bbox_max[0] - bbox_min[0])/start_vals[-2]))
       yval =np.linspace(bbox_min[1],bbox_max[1],int((bbox_max[1] - bbox_min[1])/start_vals[-2]))
       
       x, y = np.meshgrid(xval, yval)
       x, y = x.ravel(), y.ravel()
       final_points_x, final_points_y = [], []

       # calculate number inside polygon
       for i in range(len(x)):
           point = Point(x[i], y[i])
    
           if polygon.contains(point) or line.contains(point):
                total_count += 1
                final_points_x.append(x[i])
                final_points_y.append(y[i])

    return final_points_x,final_points_y

# %%
def run_SNM_contact_sim_buffer(single_ray_data, prey_stim_sizes, prey_stimuli_numbers, plot, seed): # num_stim, points_contact, points):
    '''
    Sort the eigenvalues in order of highest to lowest and assign region each 
    eigenvalue belongs to

    Parameters
    ----------
    single_ray_data : dict
        data for one of the rays eg. coordinates
    prey_size_radius : ndarray
        radius of each of the prey sizes
    prey_stimuli_numbers : ndarray
        counts for each of the stimuli for that ray

    Returns
    -------
    responses : ndarray
        Responses for each of the receptors

    '''
    random.seed(seed)
    
    num_stim = np.sum(prey_stimuli_numbers) # total number of stimuli 
    points = single_ray_data['points'] # get point
    
    num_receptors = np.size(points,0) # number of receptors on the ray
    responses = np.zeros((num_receptors, num_stim)) # pre-allocate response array
    stim_num = 0 #stim num counter
    
    # radius calculation
    prey_size_radius = prey_stim_sizes/2
        
    # go through the array for each sim size
    for i in range(len(prey_size_radius)):
        
        total_stim_radius = 0 # total stimuli of that radius calculated
        stim_cover = (())    
        
        # get ray and calculate the bounding box
        ray_poly = geometry.Polygon([[p[0], p[1]] for p in single_ray_data['boundary']])
        ray_area = single_ray_data['area']
        

        radius = prey_size_radius[i]*28 # scale sizing
    
        # load the buffer
        buffer_coords = single_ray_data['buffer_bounds'][str(prey_size_radius[i])]['coords']
        buffer_poly = geometry.Polygon(buffer_coords) # create polygon
        
        buffer_bbox = single_ray_data['buffer_bounds'][str(prey_size_radius[i])]['bbox']
        stim_xbounds, stim_ybounds = buffer_bbox[0,:], buffer_bbox[1,:]

        x_bbox_coords = np.array((stim_xbounds[0],stim_xbounds[0],stim_xbounds[1],stim_xbounds[1]))
        y_bbox_coords = np.array((stim_ybounds[0],stim_ybounds[1],stim_ybounds[0],stim_ybounds[1]))
                
        while total_stim_radius < prey_stimuli_numbers[i]: # run for stimuli
            
            # create random points
            x_rand = random.uniform(stim_xbounds[0],stim_xbounds[1])
            y_rand = random.uniform(stim_ybounds[0],stim_ybounds[1])
            buffer_coord = Point(x_rand,y_rand)

            # check the stimuli center is within the buffer shape
            if buffer_poly.contains(buffer_coord):
            
                # create stimuli
                p = geometry.point.Point(x_rand, y_rand)
                prey_stim = p.buffer(radius) # create round stim
        
                if plot: # view overlaid
                    plt.figure()
                    prey_coords = np.array(prey_stim.exterior.coords)
                    plt.plot(prey_coords[:,0],prey_coords[:,1])
                    plt.scatter(p.x,p.y,10, color='tab:red')
                    x_full,y_full = ray_poly.exterior.xy
                    plt.plot(x_full,y_full,color = 'k')
                    plt.axis('equal')
                    
                    #plot the buffer
                    plt.plot(buffer_coords[:,0],buffer_coords[:,1], color='tab:green', alpha=0.5)
                    plt.scatter(x_bbox_coords,y_bbox_coords,30,color='orange') # plot bbox
                      
                if ray_poly.intersects(prey_stim): # calculate intersection shape
                
                    inter = prey_stim.intersection(ray_poly)
                    area = inter.area # get area
                    stim_coverage = area/ ray_area # calculate as a percentage of total ray size
                    stim_cover = np.append(stim_cover,stim_coverage) # append value
                    
                    print(f"prey_radius = {prey_size_radius[i]}, curent stim = {len(stim_cover)} / {prey_stimuli_numbers[i]}")
                    
                    try: # single polygon
                        x_inter,y_inter = inter.exterior.xy
                        if plot:
                              plt.figure()
                              
                        responses_poly = check_points_buffer(ray_poly, points, inter, x_inter, y_inter, p, x_bbox_coords,y_bbox_coords, plot)
                        
                    except: # multiple polygon- run multiple poly calculation
                        print('multiple polygon')
                        responses_poly = check_points_buffer_multipoly(ray_poly, points, inter, p, x_bbox_coords,y_bbox_coords, plot)

                    responses[:,stim_num] = responses_poly # add to main array
                    stim_num += 1 # update counter for all stimuli
                    total_stim_radius += 1 # update counter for all stim radius
                    stim_cover_mean = np.mean(stim_cover)*100 # average contact size     
                    
    return stim_cover, stim_cover_mean, responses

# %%

def check_points_buffer(ray_poly, points, inter, x_inter, y_inter, p, x_bbox_coords,y_bbox_coords, plot):
    """
    sets points within ray and stimuli boundary to 1 (active).

    Parameters
    ----------
    ray_poly : Shapely polygon
        polygon of the shape.
    points : ndarray
        coordinates of each receptor point.
    inter : Shapely polygon
        intersection polygon of stimuli and ray.
    x_inter : array obj
        x coordinates of intersection polygon.
    y_inter : array obj
        y coordinates of intersection polygon.
    p : Shapely point
        coordinate of stimuli center.
    x_bbox_coords : ndarray
        bounding box coordinates x.
    y_bbox_coords :ndarray
        bounding box coordinates y.
    plot : bool
        plot response and stimuli.

    Returns
    -------
    responses_poly : ndarray
        responses of each receptor to stimulus.

    """
    responses_poly = np.zeros((len(points)))
    
    for k in range(len(points)):
        point = Point(points[k,0], points[k,1])
    
        if plot: # plot polygon
            if k == 0:
                x_full,y_full = ray_poly.exterior.xy
                plt.plot(x_full,y_full,color = 'k')
                plt.plot(x_inter,y_inter,color = 'tab:red')
                plt.scatter(x_bbox_coords,y_bbox_coords,30,color='orange')
                plt.axis('equal')
    
        inter_coords = list( np.vstack([x_inter,y_inter]).T)
        line = geometry.LineString(inter_coords)
        
        if inter.contains(point) or line.contains(point):
            responses_poly[k] = 1
            
            if plot:
                plt.scatter(points[k,0], points[k,1], 10, color= 'tab:blue')
                
    if plot: # plot the stimuli
        plt.scatter(p.x,p.y,100,color='tab:red')
    
    return responses_poly


# %%

def check_points_buffer_multipoly(ray_poly, points, inter, p, x_bbox_coords,y_bbox_coords,plot):
    """
    sets points within ray and stimuli boundary to 1 (active).
    Here the intersection between the ray and the stimulus will be
    mulitple polygons.
    
    Parameters
    ----------
    ray_poly : Shapely polygon
        polygon of the shape.
    points : ndarray
        coordinates of each receptor point.
    inter : Shapely polygon
        intersection polygon of stimuli and ray.
    x_inter : array obj
        x coordinates of intersection polygon.
    y_inter : array obj
        y coordinates of intersection polygon.
    p : Shapely point
        coordinate of stimuli center.
    x_bbox_coords : ndarray
        bounding box coordinates x.
    y_bbox_coords :ndarray
        bounding box coordinates y.
    plot : bool
        plot response and stimuli.

    Returns
    -------
    responses_poly : ndarray
        responses of each receptor to stimulus.

    """    
    responses_poly = np.zeros((len(points)))
    
    multi_polys = list(inter)

    if plot: # plot ray polygon
        plt.figure()
        x_full,y_full = ray_poly.exterior.xy
        plt.plot(x_full,y_full,color = 'k')
        plt.scatter(x_bbox_coords,y_bbox_coords,30,color='orange')
        plt.axis('equal')   
       
    for j in range(len(multi_polys)): # for each polygon run point check 

        x_inter,y_inter = multi_polys[j].exterior.xy # calc which points are inside polygon
    
        for k in range(len(points)):
            point = Point(points[k,0], points[k,1])
        
            if plot:
                if k == 0:
                    plt.plot(x_inter,y_inter,color = 'tab:red')
        
            inter_coords = list( np.vstack([x_inter,y_inter]).T)
            line = geometry.LineString(inter_coords)
            
            if inter.contains(point) or line.contains(point):
                responses_poly[k] = 1
                
                if plot:
                    plt.scatter(points[k,0], points[k,1],10, color= 'tab:blue')
        
    if plot: # plot the stimuli
        plt.scatter(p.x,p.y,100,color='tab:red')
    
    return responses_poly


# %% calculation exponentials

def normalise(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def func(x, sigma):
    return np.exp(-sigma * x)

def calc_sigmas(ray_data_sim, ray_data_center):
    """
    Calculate sigmas for each ray.

    Returns
    -------
    all_sigma : ndarray
        sigma for each ray

    """
    rays = ['Ray01','Ray02','Ray03','Ray04','Ray05','Ray06','Ray07','Ray08','Ray09','Ray10','Ray11']

    one_mm = 28 * 2 # size conversion

    # run curve fitting
    all_sigma= np.zeros(len(ray_data_sim))
    
    for i in range(len(ray_data_sim)):
    
        single_ray_data = ray_data_center[rays[i]]
        responses = ray_data_sim[rays[i]]
        out_cov =np.cov(responses)
        
        # calculate distances between receptors in region
        distance_matrix = pdist(np.vstack([single_ray_data['points'][:,0],single_ray_data['points'][:,1]]).T)
        distance_matrix = squareform(distance_matrix)/ one_mm
        plt.figure()
        plt.scatter(distance_matrix.flatten(),normalise(out_cov.flatten()))
        popt_norm, pcov_norm = curve_fit(func, distance_matrix.flatten(), normalise(out_cov.flatten()))    
        all_sigma[i] = popt_norm
        
    return all_sigma 
