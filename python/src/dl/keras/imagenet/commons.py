'''
The common files used in imagenet.
'''


import os
import sys


# finding the root of the project
current_path = os.getcwd()
python_root = 'kernelphysiology/python/'
project_dir = current_path.split(python_root, 1)[0]
python_root = os.path.join(project_dir, python_root)
sys.path += [os.path.join(python_root, 'src/')]
sys.path += ['/home/arash/Software/repositories/tensorflow/models/research/slim/']
