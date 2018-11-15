# Had to export the following before running
# export CFLAGS=-sysroot,/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.7.sdk
# export LDFLAGS=-L/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.7.sdk/usr/lib
#
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy

gco_directory = "gco_src"

files = ['GCoptimization.cpp', 'graph.cpp', 'LinkedBlockList.cpp',
         'maxflow.cpp']

files = [os.path.join(gco_directory, f) for f in files]
files.insert(0, "gco3d_python.pyx")

setup(cmdclass={'build_ext': build_ext},
      name='pygco3d',
      ext_modules=[Extension("pygco3d", files, language="c++",
                             include_dirs=[gco_directory, numpy.get_include()],
                             library_dirs=[gco_directory],
                             extra_compile_args=["-fpermissive"])])
