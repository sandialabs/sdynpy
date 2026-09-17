input_file = 'data_model_unformatted.ipynb'
output_file = 'data_model_formatted.ipynb'

import os
import inspect
import importlib
import sdynpy
import numpy as np

starting_directory = os.path.split(sdynpy.__file__)[0]

names = {}
packages = {}

for name,value in sdynpy.__dict__.items():
    if name[:2] == '__':
        continue
    if name in names:
        print('Warning: {:} is already a name!'.format(name))
    if inspect.ismodule(value):
        names['sdynpy.'+name] = (value.__name__,value)
        names['sdpy.'+name] = (value.__name__,value)
        names[name] = (value.__name__,value)
        names[value.__name__] = (value.__name__,value)
    else:
        names[name] = (value.__module__+'.'+value.__name__,value)
        names['sdynpy.'+name] = (value.__module__+'.'+value.__name__,value)
        names['sdpy.'+name] = (value.__module__+'.'+value.__name__,value)
        names[value.__module__+'.'+value.__name__] = (value.__module__+'.'+value.__name__,value)

def splitall(path):
    allparts = []
    while True:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

for root, directories, files in os.walk(starting_directory):
    files = [file for file in files if os.path.splitext(file)[-1] == '.py']
    if len(files) == 0:
        continue
    relpath = os.path.relpath(root,starting_directory)
    if relpath == '.':
        continue
    
    module_name = '.'.join(['sdynpy']+splitall(relpath))
    
    print('Importing {:}'.format(module_name))

    this_module = importlib.import_module(module_name)

    name = module_name
    value = this_module
    
    if name in names and (names[name] is None or value is not names[name][1]):
        print('Warning: {:} is already a name!'.format(name))
        names[name] = None
    else:
        names[name] = (value.__name__,value)
    names[module_name+'.'+name] = (value.__name__,value)

    for name,value in this_module.__dict__.items():
        if name[:2] == '__':
            continue
        if inspect.ismodule(value):
            if name in names and (names[name] is None or value is not names[name][1]):
                print('Warning: {:} is already a name!'.format(name))
                names[name] = None
            else:
                names[name] = (value.__name__,value)
            names[module_name+'.'+name] = (value.__name__,value)
        else:
            if name in names and (names[name] is None or value is not names[name][1]):
                print('Warning: {:} is already a name!'.format(name))
                names[name] = None
            else:
                names[name] = (value.__module__+'.'+value.__name__,value)
            names[value.__module__+'.'+value.__name__] = (value.__module__+'.'+value.__name__,value)
    
    for file in files:
        if file[:2] == '__':
            continue
        this_module_name = module_name+'.'+os.path.splitext(file)[0]
        print('Importing {:}'.format(this_module_name))
        this_module = importlib.import_module(this_module_name)

        for name,value in this_module.__dict__.items():
            if name[:2] == '__':
                continue
            if inspect.ismodule(value):
                if name in names and (names[name] is None or value is not names[name][1]):
                    print('Warning: {:} is already a name!'.format(name))
                    names[name] = None
                else:
                    names[name] = (value.__name__,value)
                names[value.__name__] = (value.__name__,value)
            else:
                if name in names and (names[name] is None or value is not names[name][1]):
                    print('Warning: {:} is already a name!'.format(name))
                    names[name] = None
                else:
                    try:
                        names[name] = (value.__module__+'.'+value.__name__,value)
                    except AttributeError:
                        names[name] = (this_module_name+'.'+name,value)
                try:
                    names[value.__module__+'.'+value.__name__] = (value.__module__+'.'+value.__name__,value)
                except AttributeError:
                    names[this_module_name+'.'+name] = (this_module_name+'.'+name,value)
                    
# Now go through each name and add the class methods
all_names = [name for name in names.keys()]
for name in all_names:
    data = names[name]
    if data is None:
        continue
    full_name,item = data
    if not isinstance(item,type):
        continue
    for member_name,member in inspect.getmembers(item):
        if member_name[0] == '_':
            continue
        if isinstance(member,property):
            for this_name in [member_name,name+'.'+member_name]:
                if this_name in names and (names[this_name] is None or member is not names[this_name][1]):
                    print('Warning: {:} is already a name!'.format(this_name))
                    names[this_name] = None
                else:
                    names[this_name] = (full_name+'.'+member_name,member)
        if inspect.isfunction(member) or inspect.ismethod(member):
            if member.__module__ is None:
                continue
            for this_name in [member_name,name+'.'+member_name]:
                if this_name in names and (names[this_name] is None or member is not names[this_name][1]):
                    print('Warning: {:} is already a name!'.format(this_name))
                    names[this_name] = None
                else:
                    names[this_name] = (member.__module__+'.'+member.__qualname__,member)

with open(input_file,'r') as fi:
    with open(output_file,'w') as fo:
        for line_index,line in enumerate(fi):
            code_offsets = np.reshape([i for i,char in enumerate(line) if char == '`' and line[i+1] == '`'],(-1,2))
            for code_offset in code_offsets[::-1]:
                code_string = line[code_offset[0]+2:code_offset[1]]
                if code_string in names:
                    if names[code_string] is None:
                        print('Could not uniquely identify code {:} at line {:} offset {:}'.format(code_string,line_index,code_offset))
                        new_code_string = '``'+code_string+'``'
                    else:
                        # See if it's a class, function, module
                        full_name,value = names[code_string]
                        if inspect.ismodule(value):
                            new_code_string = ':py:mod:`{:}<{:}>`'.format(code_string,full_name)
                        elif inspect.isclass(value):
                            new_code_string = ':py:class:`{:}<{:}>`'.format(code_string,full_name)
                        elif inspect.isfunction(value) or inspect.ismethod(value) or isinstance(value,property):
                            new_code_string = ':py:func:`{:}<{:}>`'.format(code_string,full_name)
                        else:
                            print('Could not determine type of {:} at line {:} offset {:}'.format(code_string,line_index,code_offset))
                            new_code_string = '``'+code_string+'``'
                else:
                    print('Unknown Name {:} at line {:} offset {:}'.format(code_string,line_index,code_offset))
                    new_code_string = '``'+code_string+'``'
                line = line[:code_offset[0]] + new_code_string + line[code_offset[-1]+2:]
            fo.write(line)
