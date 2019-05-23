=======
smtools
=======

Single Molecule analysis Tools
------------------------------
|pypi|  |rtd|

This is a package written for the analysis of single
molecule data by the `Redding Lab`_, at the University of Cailfornia, San Francisco.

.. _Redding Lab: https://www.reddinglab.com

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

Alternatively, the version under development lives here, and can be installed by::

    $ pip install git+https://github.com/ReddingLab/smtools

Documentation
-------------
Documentation: https://smtools.readthedocs.io/en/latest/

Usage
-----

Alignment
~~~~~~~~~

The `smtools.algnment` module is designed to align images that were split
into separate channels during data collection. It relies on using
fluorescent particles that appear in both channels. To see how it works
see our walkthrough `here`_ . Also `finding maxima`_ and `fitting gaussians`_.

.. _here: https://github.com/ReddingLab/Learning/blob/master/image-analysis-basics/4__Image-alignment-with-toolbox.ipynb
.. _finding maxima: https://github.com/ReddingLab/Learning/blob/master/image-analysis-basics/2__finding-local-maxima.ipynb
.. _fitting gaussians: https://github.com/ReddingLab/Learning/blob/master/image-analysis-basics/3__fitting-gaussians.ipynb

.. code-block:: python

    import smtools.testdata as test
    import smtools.alignment as al
    import matplotlib.pyplot as plt
    dx, dy, params = al.inspect_global_fit(test.image_stack(), showplot=False)
    im = test.image_stack()[0]
    im_old = al.overlay(im)
    im_adj_image = al.align_by_offset(im,dx,dy)
    im_new = al.overlay(im_adj_image)
    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212,sharex=ax1)
    ax1.set_title('Original Image', fontsize = "18")
    ax2.set_title('Aligned Image', fontsize = "18")
    ax1.imshow(im_old)
    ax2.imshow(im_new)
    plt.show()

.. figure:: https://github.com/ReddingLab/smtools/blob/master/images/alignment.png
   :alt:

Detecting DNA curtains
~~~~~~~~~~~~~~~~~~~~~~

The `smtools.curtains` module is designed to locate, and fit the intensity
profile of individual DNA molecules within a DNA curtain. For a description
of how it works see our walkthroughs `part 1`_, `part 2`_, and `part 3`_.

.. _part 1: https://github.com/ReddingLab/Learning/blob/master/image-analysis-basics/5__DNA_curtain_finder_1.ipynb
.. _part 2: https://github.com/ReddingLab/Learning/blob/master/image-analysis-basics/6__DNA_curtain_finder_2.ipynb
.. _part 3: https://github.com/ReddingLab/Learning/blob/master/image-analysis-basics/7__DNA_curtain_finder_3.ipynb

.. code-block:: python

    import smtools.curtains as cs
    import smtools.alignment as al
    from scipy.ndimage.interpolation import rotate
    import matplotlib.pyplot as plt
    import smtools.testdata as test
    im = test.test_curtain()
    ch1,ch2 = al.im_split(im)
    angle = cs.find_rotang(ch2)
    rotated_ch2 = rotate(ch2,angle)
    bounds, mask = cs.find_curtain(rotated_ch2)
    strands = cs.find_DNA(rotated_ch2,bounds)
    DNAs = cs.fit_DNA(rotated_ch2, strands)
    fig = plt.figure(figsize=(5,10))
    plt.axis('off')
    plt.imshow(rotated_ch2)
    for x0,x1,y in DNAs:
        plt.plot([x0,x1],[y,y],"r.", markersize = 5)
    plt.show()

.. figure:: https://github.com/ReddingLab/smtools/blob/master/images/curtain_finder.png
   :alt: 


Version History
---------------
  * 0.1.0  Initial release with `alignment`, `point_fitting` modules
  * 0.2.0  Included `curtains` and `misc` modules


References
----------
  * Github: https://github.com/ReddingLab/smtools/
  * PyPI Page: https://pypi.org/project/smtools/
  * Readthedocs: https://smtools.readthedocs.io/en/latest/


.. |pypi| image:: https://img.shields.io/pypi/v/smtools.svg
    :target: https://pypi.org/project/smtools/

.. |rtd| image:: https://img.shields.io/readthedocs/smtools.svg
    :target: https://smtools.readthedocs.io/en/latest/