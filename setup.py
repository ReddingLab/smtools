import setuptools
import os

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()

setuptools.setup(
    name="smtools",
    version="0.2.2",
    author="Sy Redding",
    description="Redding Lab analysis tools",
    long_description=read('README.rst'),
    url="https://github.com/ReddingLab/smtools",
    packages=setuptools.find_packages(),
    package_data={'smtools': ['testdata/*.tif']},
    include_package_data=True,
    install_requires=[
          'numpy','scipy','scikit-image','matplotlib'
      ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)