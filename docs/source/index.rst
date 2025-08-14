.. SDynPy documentation master file, created by
   sphinx-quickstart on Mon Jan 31 10:42:59 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: /images/logo_horizontal.svg
  :width: 550
  :alt: SDynPy Logo
  :align: center

|

Welcome to SDynPy's documentation!
==================================

|documentation| |build| |codecov| |coveralls| |codefactor| |pylint| |docker|

**SDynPy** is a package for performing structural dynamic analyses using Python.
It contains several objects that represent various structural dynamics data
types (shapes, data, geometry, etc.) as well as various functions and methods
to make working with these objects easy!

Check out the :doc:`usage` section for further information on how to use the package,
including how to :ref:`install <installation>` the package.

To quickly get running with SDynPy, check out the  out the :doc:`examples` for
an overview of recommended workflows and a summary of current functionality.

***********
Information
***********

- `Documentation <https://sandialabs.github.io/sdynpy/>`_
- `Project <https://github.com/sandialabs/sdynpy>`_
- `Releases <https://github.com/sandialabs/sdynpy/releases>`_
- `Tutorial <https://sandialabs.github.io/sdynpy/sdynpy_showcase.html>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:
    
   usage
   sdynpy_showcase
   core_functionality
   examples
   modal_tutorials
   modules
   bibliography

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

..
    Badges ========================================================================

.. |documentation| image:: https://img.shields.io/github/actions/workflow/status/sandialabs/sdynpy/pages.yml?branch=main&label=Documentation
    :target: https://sandialabs.github.io/sdynpy/

.. |build| image:: https://img.shields.io/github/actions/workflow/status/sandialabs/sdynpy/main.yml?branch=main&label=GitHub&logo=github
    :target: https://github.com/sandialabs/sdynpy

.. |pylint| image:: https://raw.githubusercontent.com/sandialabs/sdynpy/gh-pages/pylint.svg
    :target: https://github.com/sandialabs/sdynpy

.. |coveralls| image:: https://img.shields.io/coveralls/github/sandialabs/sdynpy?logo=coveralls&label=Coveralls
    :target: https://coveralls.io/github/sandialabs/sdynpy?branch=main

.. |codecov| image:: https://img.shields.io/codecov/c/github/sandialabs/sdynpy?label=Codecov&logo=codecov
    :target: https://codecov.io/gh/sandialabs/sdynpy

.. |codefactor| image:: https://img.shields.io/codefactor/grade/github/sandialabs/sdynpy?label=Codefactor&logo=codefactor
   :target: https://www.codefactor.io/repository/github/sandialabs/sdynpy

.. |docker| image:: https://img.shields.io/docker/v/dprohe/sdynpy?color=0db7ed&label=Docker%20Hub&logo=docker&logoColor=0db7ed
    :target: https://hub.docker.com/r/dprohe/sdynpy
