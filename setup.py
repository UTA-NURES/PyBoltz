from setuptools import setup, Extension
from setuptools import find_namespace_packages
from Cython.Distutils import build_ext

import os
import Cython
import numpy
from io import open
import glob

def returnPyxFiles(path):
    l = []
    for i in os.listdir(path):
        if i.endswith(".pyx") or i.endswith(".c") or i.endswith(".pxd"):
            l.append(path+i)
    return l


def returnPxdFiles(path):
    l = []
    for i in os.listdir(path):
        if i.endswith(".pxd") or i.endswith(".c"):
            l.append(path+i)
    return l

def makeExtensions(path):
    extensions = []
    for root,dirs,files in os.walk(path):
        for file in files:
            if file.endswith(".pyx"):
                moduleFiles = [] 
                moduleFiles.append(root+'/'+file)
                #if root+'/'+(file.split('.')[0]+'.pxd') in glob.glob(root+"/*.pxd"):
                    #moduleFiles.append(root+'/'+(file.split('.')[0]+'.pxd'))
                pathWithFile = root+'/'+file.split('.')[0]
                moduleName = pathWithFile.replace('/','.')
                extensions.append(Extension(moduleName,moduleFiles,
                include_dirs=[numpy.get_include(),os.getcwd()+'/PyBoltz/',os.getcwd()+'/PyBoltz/C/',os.getcwd()+'/PyBoltz/MonteFuncs/',
                os.getcwd()+'/PyBoltz/Townsend/',os.getcwd()+'/PyBoltz/Townsend/CollisionFrequencyCalc/',os.getcwd()+'/PyBoltz/Townsend/Friedland/',os.getcwd()+'/PyBoltz/Townsend/Monte/',os.getcwd()+'/PyBoltz/Townsend/PulsedTownsend/',os.getcwd()+'/PyBoltz/Townsend/SteadyStateTownsend/',os.getcwd()+'/PyBoltz/Townsend/TimeOfFlight/']))
                
    return extensions
extensions = makeExtensions('PyBoltz')
'''[
    Extension("PyBoltz",returnPyxFiles("PyBoltz/"),include_dirs=[numpy.get_include(),os.getcwd()+'/PyBoltz/',os.getcwd()+'/PyBoltz/C/']),
    Extension("PyBoltz.Townsend.Monte", returnPyxFiles("PyBoltz/Townsend/Monte/")+returnPxdFiles("PyBoltz/Townsend/Monte/"), include_dirs=[numpy.get_include(), os.getcwd()+'/PyBoltz/',os.getcwd()+'/PyBoltz/Townsend/Monte/']),
    Extension("PyBoltz.Townsend", returnPyxFiles("PyBoltz/Townsend/"), include_dirs=[numpy.get_include(), os.getcwd()+'/PyBoltz/']),
    Extension("PyBoltz.Townsend.CollisionFrequencyCalc", returnPyxFiles("PyBoltz/Townsend/CollisionFrequencyCalc/"), include_dirs=[numpy.get_include(), os.getcwd()+'/PyBoltz/']),
    Extension("PyBoltz.Townsend.PulsedTownsend", returnPyxFiles("PyBoltz/Townsend/PulsedTownsend/"), include_dirs=[numpy.get_include(), os.getcwd()+'/PyBoltz/']),
    Extension("PyBoltz.Townsend.TimeOfFlight", returnPyxFiles("PyBoltz/Townsend/TimeOfFlight/"), include_dirs=[numpy.get_include(), os.getcwd()+'/PyBoltz/']),
    Extension("PyBoltz.Townsend.SteadyStateTownsend", returnPyxFiles("PyBoltz/Townsend/SteadyStateTownsend/"), include_dirs=[numpy.get_include(), os.getcwd()+'/PyBoltz/']),
    Extension("PyBoltz.Townsend.Friedland", returnPyxFiles("PyBoltz/Townsend/Friedland/"), include_dirs=[numpy.get_include(), os.getcwd()+'/PyBoltz/']),
    Extension("PyBoltz.Monte",returnPyxFiles("PyBoltz/Monte/"),include_dirs=[numpy.get_include(),os.getcwd()+'/PyBoltz/']),
]'''

#print(returnPyxFiles("PyBoltz/Townsend/Monte/")+returnPxdFiles("PyBoltz/Townsend/Monte/"))
setup(
    setup_requires=[
        'cython>=3.0',
        'numpy',
    ],
    zip_safe=False,
    name='PyBoltz',  # Required
    packages=find_namespace_packages(),

    version='1.1.1',  # Required
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4',
    package_dir={'PyBoltz': 'PyBoltz'},
    package_data={  # Optional
        'PyBoltz': returnPxdFiles("./PyBoltz/"),
        'PyBoltz/MonteFuncs': returnPxdFiles("./PyBoltz/MonteFuncs/"),
        'PyBoltz/Townsend/Monte': returnPxdFiles("./PyBoltz/Townsend/Monte/"),
        'PyBoltz/Townsend/Friedland': returnPxdFiles("./PyBoltz/Townsend/Friedland/"),
        'PyBoltz/Townsend/SteadyStateTownsend': returnPxdFiles("./PyBoltz/Townsend/SteadyStateTownsend/"),
        'PyBoltz/Townsend/TimeOfFlight': returnPxdFiles("./PyBoltz/Townsend/TimeOfFlight/"),
        'PyBoltz/Townsend/PulsedTownsend': returnPxdFiles("./PyBoltz/Townsend/PulsedTownsend/"),
        'PyBoltz/Townsend/CollisionFrequencyCalc': returnPxdFiles("./PyBoltz/Townsend/CollisionFrequencyCalc/"),

    },

    install_requires=['numpy','cython','PyGasMix @ git+https://github.com/UTA-REST/PyGasMix.git#egg=PyGasMix-1.2.1'],  # Optional
    include_package_data = True,
    ext_modules = extensions,
    cmdclass={'build_ext': build_ext},
)

B
