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

# Create a second component a with different nodes
cubit.brick(nose_length,nose_height,depth)
cubit.cmd('move Volume 9  x {:} y 0 z 0'.format(nose_length/2.0+wing_width/2.0))
cubit.cmd('volume 9 size {:}'.format(min([nose_length,nose_height,wingspan,wing_width,depth,winglet_length,winglet_width])/(mesh_spacing+1)))
cubit.cmd('mesh volume 9')

# Set up boundary conditions
cubit.cmd('block 1 volume 1')
cubit.cmd('block 3 Volume 2 5 6 7 8')
cubit.cmd('block 2 Volume 3 4')
cubit.cmd('block 4 Volume 9')

cubit.cmd('nodeset 1 surface 4')
cubit.cmd('nodeset 2 surface 68')
cubit.cmd('draw block all nodeset all')

# Export
models = {'a':'1','c':'3','bc':'2 3','ac':'1 3','abc':'1 2 3','a2':'4'}
for label,block in models.items():
    cubit.cmd('export mesh "Z:/modal/sdynpy_substructuring/block_{:}.exo"  block {:}  overwrite '.format(label,block))
    
print('Finished!')