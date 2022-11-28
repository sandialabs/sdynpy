import cubit
cubit.reset()

nose_length = 0.5
nose_height = 0.1
wingspan = 0.5
wing_width = 0.125
depth = 0.1
winglet_length = 0.2
winglet_width = 0.1

mesh_spacing = 4

cubit.brick(nose_length,nose_height,depth)
cubit.cmd('move Volume 1  x {:} y 0 z 0'.format(nose_length/2.0+wing_width/2.0))

cubit.brick(wing_width,wingspan,depth)

for i in range(2):
    cubit.brick(winglet_length,winglet_width,depth)
    cubit.cmd('move volume {:} x {:} y {:} z 0'.format(3+i,-winglet_length/2.0-wing_width/2.0,-wingspan/2.0 + winglet_width/2 + i*(wingspan - winglet_width)))
    
# Do webcuts
webcut_distances = [-wingspan/2.0+winglet_width,-nose_height/2.0,wingspan/2.0-winglet_width,nose_height/2.0]
for distance in webcut_distances:
    cubit.cmd('webcut volume all with general plane y offset {:}'.format(distance))

cubit.cmd('merge all')

# Set mesh spacing
cubit.cmd('volume all size {:}'.format(min([nose_length,nose_height,wingspan,wing_width,depth,winglet_length,winglet_width])/mesh_spacing))
cubit.cmd('mesh volume all')

# Set up boundary conditions
cubit.cmd('block 1 volume 1')
cubit.cmd('block 2 volume all except volume in block 1')

cubit.cmd('nodeset 1 surface 4')
cubit.cmd('draw block all nodeset all')

# Export
for block in [1,2,'all']:
    cubit.cmd('export mesh "Z:/python_utilities/sdynpy/test_scripts/CB_Substructuring/block_{:}.exo"  block {:}  overwrite '.format(block,block))
    
print('Finished!')