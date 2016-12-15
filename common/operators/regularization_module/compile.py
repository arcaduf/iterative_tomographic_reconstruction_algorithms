import os
import shutil


if os.path.exists('build'):
    shutil.rmtree('build')

if os.path.isfile('penalty.so'):
    os.remove('penalty.so')

if os.path.isfile('penalty.c'):
    os.remove('penalty.c') 

os.system('python create_penalty_module.py build_ext --inplace')
