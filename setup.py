from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='MolDistill',
    version='0.0.2',
    packages=find_packages(),
    url='',
    license='MIT',
    author='Myself',
    author_email='my@mail.com',
    description='',
)