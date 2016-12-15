import os
import shutil


if os.path.exists('build'):
    shutil.rmtree('build')

if os.path.isfile( 'split_bregman_nltv.c' ) is True:
    os.remove( 'split_bregman_nltv.c' ) 

if os.path.isfile( 'split_bregman_nltv.so' ) is True:
    os.remove( 'split_bregman_nltv.so' )
