"""
Setup script for lunapy - Python bindings for LUNA
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """Extension that is built using CMake"""
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build extension that runs CMake"""

    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build lunapy")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # CMake configure arguments
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DBUILD_PYTHON_BINDINGS=ON',
            '-DBUILD_TESTS=OFF',  # Don't build tests when installing Python package
        ]

        # Build configuration
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']

        # Parallel build
        if hasattr(self, 'parallel') and self.parallel:
            build_args += [f'-j{self.parallel}']
        else:
            build_args += ['-j4']

        # Build directory
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Run CMake configure
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=self.build_temp
        )

        # Run CMake build
        subprocess.check_call(
            ['cmake', '--build', '.', '--target', 'LunaPyCore'] + build_args,
            cwd=self.build_temp
        )


# Read README
readme_path = Path(__file__).parent / 'lunapy' / 'README.md'
if readme_path.exists():
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = 'Python bindings for LUNA neural network bound propagation'

setup(
    name='lunapy',
    version='1.0.0',
    author='LUNA Team',
    description='Python bindings for LUNA neural network bound propagation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/luna',
    packages=['lunapy'],
    ext_modules=[CMakeExtension('lunapy.LunaPyCore')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.19.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.10',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: C++',
    ],
    keywords='neural-networks verification bounds-propagation CROWN',
)
