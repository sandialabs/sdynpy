# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:52:28 2023

@author: dprohe
"""

import numpy as np
import sdynpy as sdpy
import matplotlib.pyplot as plt
plt.xkcd()
plt.close('all')

# Create a beam model so we can get some FRFs

system,geometry = sdpy.System.beam(1,0.15,0.1,500,material='steel')

modes = system.eigensolution(num_modes=12)
modes.damping = 0.002

response_dofs = sdpy.coordinate_array(geometry.node.id,3)
reference_dofs = sdpy.coordinate_array([1],3)



#%% 
plt.close('all')
frfs = modes.compute_frf(np.arange(9000)/10+1,response_dofs,reference_dofs).flatten()
frfs = frfs[1:20:3]
# gp = sdpy.GUIPlot(frfs)

# frfs.plot()

cam_data = np.load('camera.npz')
K = cam_data['K']
RT = cam_data['RT']

frf_x = []
frf_y = []
# Move the FRFs into a block that is 2 x 2 x 2
yscale = 2/frfs.abscissa.max()
log_ordinate = np.log(np.abs(frfs.ordinate))
zscale = 2/(log_ordinate.max()-log_ordinate.min())
zshift = log_ordinate.min()
figure,axis = plt.subplots(num='logo2', figsize=[1.96, 1.83])
for i,frf in enumerate(frfs):
    x = -1+(i/(frfs.size-1))*2
    print(x)
    y = frf.abscissa*yscale-1
    z = np.log(np.abs(frf.ordinate))
    z = (z - zshift)*zscale-1
    # Construct some arrays
    array = np.array(np.broadcast_arrays(x,y,z))
    img_coords = sdpy.camera.compute_pixel_position(K, RT, array)
    axis.plot(*img_coords,color='#4472C4')
axis.set_aspect('equal')
axis.set_xlim([0,1080])
axis.set_ylim([1080,0])
axis.axis('off')
# figure.savefig('logo.png')
# figure.savefig('logo.pdf')