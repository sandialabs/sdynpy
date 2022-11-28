cubit.reset()
cmd = cubit.cmd

cube_size = [1.0,1.2,1.5]
cube_thickness = 0.1
lid_depth = 0.15

component_size = [0.4,0.4,0.7]

# Create the box
cmd('brick x {:} y {:} z {:}'.format(*cube_size))
cmd('brick x {:} y {:} z {:}'.format(*[v-cube_thickness*2 for v in cube_size]))
cmd('subtract volume 2 from volume 1')
cmd('compress ids')
cmd('webcut volume 1 with plane zplane offset {:}'.format(cube_size[-1]/2-lid_depth))

# Create the component on the lid of the box
cmd('brick x {:} y {:} z {:}'.format(*component_size))
cmd('move Volume 3  z {:} include_merged '.format(cube_size[-1]/2-cube_thickness-component_size[-1]/2))

cmd('imprint all')
cmd('merge all')

cmd('volume all size {:}'.format(cube_thickness/3))
cmd('mesh volume all')

cmd('block 1 volume 2')
cmd('block 2 volume 1')
cmd('block 3 volume 3')

cmd('export mesh "a.exo"  block 1  overwrite')
cmd('export mesh "b.exo"  block 2  overwrite')
cmd('export mesh "c.exo"  block 3  overwrite')
cmd('export mesh "ab.exo"  block 1 2 overwrite')
cmd('export mesh "bc.exo"  block 2 3 overwrite')
cmd('export mesh "abc.exo"  block 1 2 3  overwrite')