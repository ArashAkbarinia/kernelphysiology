'''
The common files used in stl.
'''


import os
import sys


# finding the root of the project
current_path = os.getcwd()
python_root = 'kernelphysiology/python/'
project_dir = current_path.split(python_root, 1)[0]
python_root = os.path.join(project_dir, python_root)
project_dir = os.path.join(project_dir, 'kernelphysiology')
sys.path += [os.path.join(python_root, 'src/')]
