#experience_replay_setup.py
from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('experience_replay.pyx'))