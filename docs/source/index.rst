.. alignment_tools documentation master file, created by
   sphinx-quickstart on Sun May 27 17:40:25 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

smtools: The Redding Lab analysis package
=========================================
This is a package written for the analysis of single
molecule data by the `Redding Lab`_, at the University of Cailfornia, San Francisco.

.. _Redding Lab: https://www.reddinglab.com


.. toctree::
   :maxdepth: 1
   :caption: PACKAGE CONTENTS:

   point_fitting
   alignment
   curtains
   stacks
   misc_module


Dependencies
------------
  * numpy
  * scipy
  * scikit-image
  * matplotlib

.. _numpy: http://www.numpy.org/
.. _scipy: https://www.scipy.org/
.. _scikit-image: https://scikit-image.org/
.. _matplotlib: https://matplotlib.org/

Installation
------------

The easiest way to install the package is via ``pip``::

    $ pip install smtools

Alternatively, the version under development can be installed by::

    $ pip install git+https://github.com/ReddingLab/smtools

Documentation
-------------
Documentation: https://smtools.readthedocs.io/en/latest/


Version History
---------------

0.3.0
~~~~~
* Addition of the `Curtain` class.
* Addition of the `stacks` module
* Major changes to `point_fitting.fit_routine`. Now does error filtering.
  Now accepts and returns lists of (x,y) tuples rather than single tuples.
* Reorganization of the documentation.

0.2.0
~~~~~
* Addition of `curtains` and `misc` modules

0.1.0
~~~~~
* Initial release with `alignment`, `point_fitting` modules


References
----------
  * Github: https://github.com/ReddingLab/smtools/
  * PyPI Page: https://pypi.org/project/smtools/
  * Readthedocs: https://smtools.readthedocs.io/en/latest/




