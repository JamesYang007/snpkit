from glob import glob
from setuptools import setup, find_packages
from distutils.dir_util import copy_tree
from pybind11.setup_helpers import Pybind11Extension 
from pybind11.setup_helpers import ParallelCompile
import sysconfig
import os
import platform
import subprocess


def run_cmd(cmd):
    try:
        output = subprocess.check_output(
            cmd.split(" "), stderr=subprocess.STDOUT
        ).decode()
    except subprocess.CalledProcessError as e:
        output = e.output.decode()
        raise RuntimeError(output)
    return output.rstrip()


# Optional multithreaded build
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

__version__ = open("VERSION", "r").read()

# ENVPATH = os.getenv("CONDA_PREFIX")

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-Wall", "-Wextra", "-DNDEBUG", "-O3"]
libraries = []
extra_link_args = []

system_name = platform.system()
if (system_name == "Darwin"):
    omp_prefix = run_cmd("brew --prefix libomp")
    omp_include = os.path.join(omp_prefix, "include")
    omp_lib = os.path.join(omp_prefix, "lib")
    extra_compile_args += [
        f"-I{omp_include}",
        "-Xclang",
        "-fopenmp",
    ]
    extra_link_args += [f'-L{omp_lib}']
    libraries = ['omp']
    
if (system_name == "Linux"):
    extra_compile_args += ["-fopenmp"]
    libraries = ['gomp']

ext_modules = [
    Pybind11Extension(
        "snpkit.snpkit_core",
        sorted(glob("snpkit/src/*.cpp")),  # Sort source files for reproducibility
        defines=[],
        include_dirs=[
            "snpkit/src",
            "snpkit/src/include",
            "snpkit/src/third_party/eigen3",
            # os.path.join(ENVPATH, 'include'),
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
        cxx_std=14,
    ),
]

setup(
    name='snpkit', 
    version=__version__,
    description='A Python package for SNP-related tools.',
    long_description='',
    author='James Yang',
    author_email='jamesyang916@gmail.com',
    maintainer='James Yang',
    maintainer_email='jamesyang916@gmail.com',
    packages=["snpkit"], 
    package_data={
        "snpkit": [
            "src/**/*.hpp", 
            "src/third_party/**/*",
            "snpkit_core.cpython*",
        ],
    },
    ext_modules=ext_modules,
    zip_safe=False,
)