import os
import shutil


if os.path.exists('build'):
    shutil.rmtree('build')

if os.path.isfile('split_bregman_nltv.so'):
    os.remove('split_bregman_nltv.so')

if os.path.isfile('split_bregman_nltv.c'):
    os.remove('split_bregman_nltv.c') 

os.system('python create_penalty_module.py build_ext --inplace')
