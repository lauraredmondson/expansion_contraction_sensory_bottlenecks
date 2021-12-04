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

import numpy as np
import pickle as pk
import sensory_bottlenecks as sb

# %% Activation ratio

# Data from Catania (1990) paper
usage = np.array(([15.15, 7.29, 7.72, 7.46, 7.97, 4.66, 8.98, 12.00, 19.17, 44.61, 60.61])) # usage of each ray (raw data)
total_touches = 500 # Total touches higher as data only includes prey detection.
touch = usage/ total_touches
var_ratios = touch*(1 - touch)

print('Variance ratio:')
[print(f"Ray {idx+1}: {var:.2f}") for idx,var in enumerate(var_ratios)]

# %% Sigma parameter- create stimuli set

# build half stimuli-size ray buffers and tile with points for simulation
sb.build_ray_buffers(plot=True)

# load shape data for each ray- each ray has been pre-rotated to find the bounding box.
with open('snm_params_ray_center_buffer.pk', 'rb') as handle:
  ray_data_center = pk.load(handle)  
  
# sort the above data into a dictionary
rays = ['Ray01','Ray02','Ray03','Ray04','Ray05','Ray06','Ray07','Ray08','Ray09','Ray10','Ray11']

prey_stim_sizes = np.array(([1,1.5,2,2.5,3,3.5,4,4.5]))

# array of the number of prey stim on each ray and their sizes
prey_numbers = np.array(([18,	30,	 24,	30,	  28,   25,  24,   25,	 25,	35,	  665],
                         [18,	30,	 30,	35,	  37,	35,	 27,   35,	 46,	98,	  650],
                         [18,	40,	 54,	53,	  50,	40,	 53,   60,	 97,	595,  670],
                         [76,	47,	 72,	76,	  76,	82,	 85,   93,	 133,	603,  667],
                         [106,	82,	 104,	108,  108,	116, 113,  133,	 596,	612,  673],
                         [512,	75,	 90,	90,	  90,	97,	 109,  541,	 567,	590,  635],
                         [588,	150, 567,	572,  572,	567, 581,  588,	 641,	653,  692],
                         [598,	558, 573,	572,  573,	577, 584,  606,	 650,	657,  699]))

# run the SNM simulation
all_responses = {}
prey_numbers *= 10 # increase number of stimulations for each
plot = False # whether to plot each stimuli on the rays
for idx in range(len(rays)):
    prey_stimuli_numbers = prey_numbers[:,idx]
    single_ray_data = ray_data_center[rays[idx]]
    all_responses[rays[idx]] = sb.run_SNM_contact_sim_buffer(single_ray_data,  prey_stim_sizes, prey_stimuli_numbers, plot, 0)[2]
    
# save responses
with open('snm_params_responses_buffer.pk', 'wb') as handle:
    pk.dump(all_responses, handle)
    
# %% Sigma parameter

# load ray data center
with open('snm_params_ray_center_buffer.pk', 'rb') as handle:
    ray_data_center = pk.load(handle)

# load stimuli dropping data
with open('snm_params_responses_buffer.pk', 'rb') as handle:
    ray_data_sim = pk.load(handle)

ray_sigmas = sb.calc_sigmas(ray_data_sim, ray_data_center)


